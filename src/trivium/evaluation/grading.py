"""Deterministic answer matching with an optional LLM judge fallback."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)

_STOPWORDS = re.compile(r"\b(the|a|an|of|is|was|were|are)\b")
_PUNCT = re.compile(r"[^\w\s.%-]")
# standalone numbers only: "9" in "qm9x" or "6-31g" is not a number token
_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?(?![a-z\d])", re.I)
_BOOL_TRUE = {"true", "t", "yes", "correct"}
_BOOL_FALSE = {"false", "f", "no", "incorrect"}
_HEDGE = re.compile(
    r"\b(or|not|isn't|wasn't|never|neither|nor|either|except|rather than|instead of)\b"
)
_LETTERS = "abcdefgh"


def _basic_normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:\-]?\s+", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators
    s = s.replace("%", " percent")
    s = _PUNCT.sub(" ", s)
    return " ".join(s.split()).rstrip(".")


def normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation and filler words.

    Single-token answers (option letters, "a", numbers) are returned as-is so that stop-word
    removal cannot erase them.
    """
    base = _basic_normalize(s)
    if len(base.split()) <= 1:
        return base
    stripped = " ".join(_STOPWORDS.sub(" ", base).split())
    return stripped or base


def _numbers(s: str) -> list[float]:
    out = []
    for m in _NUM.findall(s):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _numeric_parts(s: str) -> tuple[float, str] | None:
    """(value, unit) when ``s`` is essentially a number with an optional unit, else None."""
    m = re.fullmatch(
        r"\s*(?:about|approximately|~|roughly)?\s*([-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?)\s*([a-z%°µ/²³^\-\s]*)\s*",
        s,
        re.I,
    )
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = " ".join(m.group(2).lower().replace("percent", "%").split())
    return value, unit


def _numbers_close(p: float, r: float, rel_tol: float) -> bool:
    if r == p:
        return True
    if r == 0:
        return abs(p) < 1e-9
    # integers (years, counts, ranks) must match exactly; the relative tolerance is for measured
    # quantities such as 0.1201 vs 0.12
    return (not float(r).is_integer()) and abs(p - r) / abs(r) <= rel_tol


def _units_compatible(pu: str, ru: str) -> bool:
    if not ru or not pu:
        return True  # reference or prediction unitless → do not penalise
    pu, ru = pu.rstrip("s"), ru.rstrip("s")
    return pu == ru or pu.startswith(ru) or ru.startswith(pu)


def match_answer(
    prediction: str,
    references: list[str],
    answer_type: str = "short_answer",
    *,
    rel_tol: float = 0.01,
) -> bool:
    """Deterministic check that ``prediction`` matches one of ``references``.

    * boolean: first token decides.
    * choice: an option letter, or the text of the correct option.
    * numeric: the prediction must be essentially a number (optional unit); integers exact,
      others within ``rel_tol``; units must be compatible when both sides carry one.
    * short_answer: exact match after normalisation, or the reference contained in a short
      prediction without hedges/negations; numeric references use the numeric path.
    """
    pred = normalize_answer(prediction)
    if not pred:
        return False
    if answer_type == "boolean":
        first = pred.split()[0]
        ref = normalize_answer(references[0]) if references else ""
        return (first in _BOOL_TRUE and ref in _BOOL_TRUE) or (
            first in _BOOL_FALSE and ref in _BOOL_FALSE
        )
    if answer_type == "choice":
        letters = {normalize_answer(r) for r in references if len(normalize_answer(r)) == 1}
        first_tok = pred.split()[0].rstrip(")")
        if first_tok in letters and first_tok in _LETTERS:
            return True
        # fall through to text comparison against the option text(s)

    hedged = bool(_HEDGE.search(pred))
    for ref in references:
        nref = normalize_answer(ref)
        if not nref:
            continue
        if pred == nref:
            return True
        if not hedged and len(nref) > 2:
            contained = f" {nref} " in f" {pred} " and len(pred) <= 2 * len(nref) + 12
            # a correct answer followed by a qualifier: "DFT with the wB97X functional"
            qualified = pred.startswith(nref + " ") and len(pred) <= 60
            if contained or qualified:
                return True

    if answer_type in ("numeric", "short_answer"):
        for ref in references:
            rparts = _numeric_parts(normalize_answer(ref))
            if rparts is None:
                continue
            rval, runit = rparts
            pparts = _numeric_parts(pred)
            if pparts is not None:
                pval, punit = pparts
                if _numbers_close(pval, rval, rel_tol) and _units_compatible(punit, runit):
                    return True
                continue
            # prose prediction ("QM9X covers molecules with up to 9 heavy atoms"): accept when it
            # contains exactly one number, that number matches, and nothing is hedged
            pnums = _numbers(pred)
            if len(pnums) == 1 and not hedged and _numbers_close(pnums[0], rval, rel_tol):
                if not runit or re.search(rf"\b{re.escape(runit.rstrip('s'))}", pred):
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
