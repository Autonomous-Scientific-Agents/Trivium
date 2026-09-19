"""Deterministic answer matching with an optional LLM judge fallback."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)

_ARTICLES = re.compile(r"\b(the|a|an|of|is|was|were|are)\b")
_PUNCT = re.compile(r"[^\w\s.%-]")
_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?", re.I)
_BOOL_TRUE = {"true", "t", "yes", "correct"}
_BOOL_FALSE = {"false", "f", "no", "incorrect"}


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:\-]?\s+", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = s.replace("%", " percent")
    return " ".join(s.split()).rstrip(".")


def _numbers(s: str) -> list[float]:
    out = []
    for m in _NUM.findall(s):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def match_answer(
    prediction: str,
    references: list[str],
    answer_type: str = "short_answer",
    *,
    rel_tol: float = 0.01,
) -> bool:
    pred = normalize_answer(prediction)
    if not pred:
        return False
    if answer_type == "boolean":
        first = pred.split()[0] if pred.split() else ""
        ref = normalize_answer(references[0]) if references else ""
        return (first in _BOOL_TRUE and ref in _BOOL_TRUE) or (
            first in _BOOL_FALSE and ref in _BOOL_FALSE
        )
    if answer_type == "choice":
        letters = {normalize_answer(r)[:1] for r in references if len(normalize_answer(r)) == 1}
        first_tok = pred.split()[0].rstrip(")") if pred.split() else ""
        if first_tok in letters:
            return True
    for ref in references:
        nref = normalize_answer(ref)
        if not nref:
            continue
        if pred == nref:
            return True
        if len(nref) > 2 and (f" {nref} " in f" {pred} " and len(pred) <= 2 * len(nref) + 12):
            return True
    if answer_type in ("numeric", "short_answer"):
        pnums = _numbers(pred)
        for ref in references:
            rnums = _numbers(ref)
            if rnums and pnums:
                r = rnums[0]
                p = pnums[0]
                if (
                    r == p
                    or (r != 0 and abs(p - r) / abs(r) <= rel_tol)
                    or (r == 0 and abs(p) < 1e-9)
                ):
                    if answer_type == "numeric" or len(pnums) == 1:
                        return True
    return False


class JudgeVerdict(BaseModel):
    equivalent: bool
    explanation: str = ""


JUDGE_SYSTEM = "You grade short answers for semantic equivalence."
JUDGE_INSTRUCTIONS = """Decide whether the candidate answer means the same as (or is a strictly more specific form of) the
reference answer for this question. Different wording, unit formatting, or abbreviations are fine;
a different entity, number or sign is not. Hedged answers listing several alternatives are wrong."""


async def judge_equivalent(llm: LLM, question: str, prediction: str, references: list[str]) -> bool:
    msgs = build_messages(
        JUDGE_SYSTEM,
        JUDGE_INSTRUCTIONS,
        {"question": question, "candidate_answer": prediction, "reference_answers": references},
    )
    try:
        resp = await llm.generate(msgs, JudgeVerdict, task="judge_answer")
        return bool(resp.parsed.equivalent)  # type: ignore[union-attr]
    except LLMError as exc:
        log.warning("judge failed: %s", exc)
        return False


async def is_correct(
    prediction: str, references: list[str], answer_type: str, *, question: str, judge: LLM | None
) -> bool:
    if match_answer(prediction, references, answer_type):
        return True
    if judge is None or answer_type in ("boolean", "choice"):
        return False
    return await judge_equivalent(judge, question, prediction, references)
