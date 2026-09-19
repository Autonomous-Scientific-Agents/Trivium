"""Closed-book and open-book answering trials (design §13–14) shared by difficulty tiers."""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from trivium.evaluation.grading import is_correct
from trivium.generation.base import Candidate
from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)


class ShortAnswer(BaseModel):
    answer: str


CLOSED_SYSTEM = "You answer quiz questions from memory as accurately as you can."
CLOSED_INSTRUCTIONS = """Answer the question in at most {max_words} words. You have no reference material: rely on your own
knowledge and give your single best guess even if unsure (do not answer "unknown" or list alternatives).
For true/false questions answer "True" or "False"; for multiple choice answer with the option letter."""

OPEN_SYSTEM = "You answer quiz questions using only the provided evidence."
OPEN_INSTRUCTIONS = """Answer the question in at most {max_words} words using ONLY the evidence passages. For true/false
questions answer "True" or "False"; for multiple choice answer with the option letter. If the evidence
does not contain the answer, answer "not in evidence"."""


def _format_question(cand: Candidate) -> str:
    if cand.options:
        letters = "ABCDEFGH"
        opts = "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(cand.options))
        return f"{cand.question}\n{opts}"
    return cand.question


async def _trial(
    llm: LLM, cand: Candidate, evidence: list[str] | None, trial: int, max_words: int
) -> str | None:
    if evidence is None:
        msgs = build_messages(
            CLOSED_SYSTEM,
            CLOSED_INSTRUCTIONS.format(max_words=max_words),
            {"question": _format_question(cand)},
        )
        task = "closed_book"
    else:
        msgs = build_messages(
            OPEN_SYSTEM,
            OPEN_INSTRUCTIONS.format(max_words=max_words),
            {"question": _format_question(cand), "evidence": evidence},
        )
        task = "open_book"
    try:
        resp = await llm.generate(
            msgs, ShortAnswer, task=task, trial=trial, temperature=0.7 if trial else None
        )
        return resp.parsed.answer  # type: ignore[union-attr]
    except LLMError as exc:
        # None marks an infrastructure failure; the caller decides how to treat it. Returning ""
        # would count an outage as a wrong answer and distort accuracies.
        log.warning("%s trial %d failed: %s", task, trial, exc)
        return None


async def accuracy(
    llm: LLM,
    cand: Candidate,
    *,
    evidence: list[str] | None,
    trials: int,
    judge: LLM | None,
    max_words: int = 12,
) -> tuple[float, list[str]]:
    """Fraction of trials whose answer matches the reference; also returns the raw answers."""
    trials = max(1, trials)
    raw = await asyncio.gather(*(_trial(llm, cand, evidence, t, max_words) for t in range(trials)))
    answers = [a for a in raw if a is not None]
    if not answers:
        raise LLMError(
            f"all {trials} {'open' if evidence else 'closed'}-book trials failed for {cand.id}"
        )
    if len(answers) < trials:
        log.warning(
            "%d/%d trials failed for %s; accuracy uses the rest",
            trials - len(answers),
            trials,
            cand.id,
        )
    refs = cand.all_answers()
    verdicts = await asyncio.gather(
        *(
            is_correct(a, refs, cand.answer_type, question=cand.question, judge=judge)
            for a in answers
        )
    )
    return sum(verdicts) / len(answers), answers
