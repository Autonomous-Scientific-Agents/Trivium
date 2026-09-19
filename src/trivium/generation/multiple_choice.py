"""Convert an accepted short-answer question into multiple choice with validated distractors (§10.8)."""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from trivium.config import QuestionType
from trivium.facts.store import norm
from trivium.generation.base import Candidate, GenerationContext, answers_match
from trivium.models.llm import LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)


class Distractors(BaseModel):
    distractors: list[str] = Field(default_factory=list)


class DistractorCheck(BaseModel):
    also_correct: bool
    plausible: bool = True


DISTRACTOR_SYSTEM = "You design multiple-choice distractors."
DISTRACTOR_INSTRUCTIONS = """Propose {k} distractors for the question. Each must have the same semantic type and similar
form as the correct answer (e.g. numbers with the same unit and magnitude, or names of the same kind),
be plausible to someone who has not read the source, and be clearly wrong given the evidence.
Prefer entities/values that appear in `nearby_values` when they fit."""

CHECK_SYSTEM = "You audit multiple-choice options."
CHECK_INSTRUCTIONS = """Given the question, the evidence and the reference answer, decide whether the proposed option could
also be considered a correct answer (`also_correct`) and whether it is plausible as a distractor."""


class MultipleChoiceConverter:
    def __init__(self, ctx: GenerationContext, n_options: int = 4):
        self.ctx = ctx
        self.n_options = n_options

    def _nearby_values(self, cand: Candidate) -> list[str]:
        store = self.ctx.store
        facts = store.many(cand.source_fact_ids)
        vals: list[str] = []
        for f in facts:
            for g in store.facts_for_predicate(f.predicate):
                v = g.object.display()
                if (
                    not answers_match(v, cand.answer)
                    and v not in vals
                    and (g.object.is_numeric == f.object.is_numeric)
                ):
                    vals.append(v)
            for g in store.facts_for_subject(f.subject):
                v = g.object.display()
                if (
                    not answers_match(v, cand.answer)
                    and v not in vals
                    and (g.object.is_numeric == f.object.is_numeric)
                ):
                    vals.append(v)
        return vals[:12]

    async def convert(self, cand: Candidate, evidence: list[str]) -> Candidate | None:
        k = self.n_options - 1
        payload = {
            "question": cand.question,
            "correct_answer": cand.answer,
            "evidence": evidence,
            "nearby_values": self._nearby_values(cand),
        }
        try:
            resp = await self.ctx.llm.generate(
                build_messages(DISTRACTOR_SYSTEM, DISTRACTOR_INSTRUCTIONS.format(k=k + 2), payload),
                Distractors,
                task="distractors",
            )
        except LLMError as exc:
            log.warning("distractor generation failed: %s", exc)
            return None
        proposed = []
        for d in resp.parsed.distractors:  # type: ignore[union-attr]
            d = d.strip()
            if (
                d
                and not any(answers_match(d, a) for a in cand.all_answers())
                and norm(d) not in {norm(p) for p in proposed}
            ):
                proposed.append(d)
        checks = await asyncio.gather(*(self._check(cand, d, evidence) for d in proposed))
        valid = [d for d, ok in zip(proposed, checks, strict=True) if ok][:k]
        if len(valid) < k:
            return None
        options = [cand.answer, *valid]
        self.ctx.rng.shuffle(options)
        letters = "ABCDEFGH"
        correct_idx = options.index(cand.answer)
        mc = cand.model_copy(
            update={
                "question_type": QuestionType.multiple_choice,
                "answer_type": "choice",
                "options": options,
                "answer": letters[correct_idx],
                "accepted_answers": [
                    cand.answer,
                    f"{letters[correct_idx]}) {cand.answer}",
                    *cand.accepted_answers,
                ],
                "derivation": {
                    **(cand.derivation or {}),
                    "base_answer": cand.answer,
                    "distractors": valid,
                },
            }
        )
        return mc

    async def _check(self, cand: Candidate, distractor: str, evidence: list[str]) -> bool:
        payload = {
            "question": cand.question,
            "reference_answer": cand.answer,
            "option": distractor,
            "evidence": evidence,
        }
        try:
            resp = await self.ctx.llm.generate(
                build_messages(CHECK_SYSTEM, CHECK_INSTRUCTIONS, payload),
                DistractorCheck,
                task="check_distractor",
            )
        except LLMError:
            return False
        v: DistractorCheck = resp.parsed  # type: ignore[assignment]
        return (not v.also_correct) and v.plausible
