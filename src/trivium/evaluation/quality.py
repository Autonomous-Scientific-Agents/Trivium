"""Multi-metric LLM quality rubric (design §12)."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from trivium.facts.store import norm
from trivium.generation.base import Candidate
from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)


class QualityScores(BaseModel):
    answerability: float = Field(ge=0, le=1)
    evidence_entailment: float = Field(ge=0, le=1)
    ambiguity: float = Field(ge=0, le=1)
    answer_conciseness: float = Field(ge=0, le=1)
    clarity: float = Field(ge=0, le=1)
    answer_leakage: float = Field(ge=0, le=1)
    specificity: float = Field(ge=0, le=1)
    interestingness: float = Field(ge=0, le=1)
    notes: str = ""


QUALITY_SYSTEM = "You are a rigorous reviewer of quiz / benchmark questions."
QUALITY_INSTRUCTIONS = """Score the question independently on each metric in [0, 1]:
- answerability: the question can be answered from the evidence alone.
- evidence_entailment: the evidence implies the reference answer is correct and unique.
- ambiguity: several different answers could reasonably be considered correct (LOWER is better).
- answer_conciseness: the reference answer is short and deterministic enough for exact-match grading.
- clarity: the question is unambiguous and self-contained without unnecessary complexity.
- answer_leakage: the wording of the question gives away the answer (LOWER is better). The
  "True or False:" prefix of a true/false item and the listed options of a multiple-choice item
  are NOT leakage.
- specificity: the question targets a specific fact rather than a broad summary.
- interestingness: how engaging the question would be for a knowledgeable trivia audience.
Be strict: a question that says "according to the text" or asks "what did the authors find" should
score low on clarity/specificity."""


def heuristic_leakage(cand: Candidate) -> float:
    q = norm(cand.question)
    hits = 0
    for a in cand.all_answers():
        na = norm(a)
        if len(na) >= 3 and na in q:
            hits += 1
    return 1.0 if hits else 0.0


async def score_quality(llm: LLM, cand: Candidate, evidence: list[str]) -> QualityScores | None:
    payload = {
        "question": cand.question,
        "reference_answer": cand.answer,
        "accepted_answers": cand.accepted_answers,
        "options": cand.options,
        "evidence": evidence,
    }
    try:
        resp = await llm.generate(
            build_messages(QUALITY_SYSTEM, QUALITY_INSTRUCTIONS, payload),
            QualityScores,
            task="quality",
        )
    except LLMError as exc:
        log.warning("quality scoring failed for %s: %s", cand.id, exc)
        return None
    return resp.parsed  # type: ignore[return-value]
