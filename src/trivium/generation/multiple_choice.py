"""Convert an accepted short-answer question into multiple choice with validated distractors (§10.8).

Good distractors are what make a multiple-choice item tricky. Generic same-category guesses
("DFT Lab", "Institution Y") are eliminated at a glance. This converter builds distractors from
typed sources, most deceptive first:

* ``in_document`` – values or entities that play the *same role* in the source: the competing
  method, the value under another condition, the other compared subject. They appear in the
  evidence, so skimming does not help.
* ``misconception`` – wrong answers the closed-book model actually gave for this question.
* ``near_miss`` – for numbers: same unit and precision, adjacent magnitude, a factor-of-ten slip.
* ``sibling_property`` / ``other`` – LLM-proposed alternatives of the same type.

Every distractor is checked for type, plausibility and not being correct. The option set is then
checked for format giveaways (the correct answer is much longer or formatted differently) and by
an *options-only* probe: a model that sees only the options must not find the answer.
"""

from __future__ import annotations

import asyncio
import logging
import re

from pydantic import BaseModel, Field

from trivium.config import QuestionType
from trivium.facts.models import fmt_number
from trivium.facts.store import norm
from trivium.generation.base import Candidate, GenerationContext, answers_match
from trivium.generation.true_false import is_year_like, perturb_number
from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)

KINDS = ("in_document", "misconception", "near_miss", "sibling_property", "other")
_KIND_RANK = {k: i for i, k in enumerate(KINDS)}
_NON_ANSWERS = {"unknown", "not in evidence", "none", "n/a", "i don't know", ""}
_NUM = re.compile(r"^\s*([-+]?\d[\d,]*\.?\d*)\s*(.*?)\s*$")


class DistractorProposal(BaseModel):
    text: str
    kind: str = Field(
        "other", description="in_document | misconception | near_miss | sibling_property | other"
    )


class Distractors(BaseModel):
    distractors: list[DistractorProposal] = Field(default_factory=list)


class DistractorCheck(BaseModel):
    also_correct: bool
    plausible: bool = True
    same_type: bool = True


class OptionsOnlyPick(BaseModel):
    answer: str


DISTRACTOR_SYSTEM = "You design deceptive but fair multiple-choice distractors."
DISTRACTOR_INSTRUCTIONS = """Propose {k} distractors for the question. The goal is options that a reader who skimmed the
source, or who relies on general knowledge, would plausibly pick. In order of preference:
- "in_document": an entity or value that plays the same role in the source (the competing method,
  the value under another condition, another compared subject). Candidates are in
  `pools.in_document`.
- "misconception": a wrong answer people (or models) actually give; see `pools.misconception`.
- "near_miss": for numbers, same unit and precision with an adjacent magnitude or a common slip
  (factor of ten, swapped values); see `pools.near_miss`.
- "sibling_property": something true of a closely related entity but not of this one.
- "other": any other plausible alternative of the same type.
Every distractor must have the same semantic type as the correct answer (a year for a year, a
person for a person, the same unit for a quantity), be clearly wrong given the evidence, and match
the correct answer's form: similar length, specificity, capitalisation and numeric precision. Never
use placeholders ("Institution Y", "Method A"), "all/none of the above", or joke answers. Tag each
distractor with its `kind`."""

CHECK_SYSTEM = "You audit multiple-choice options."
CHECK_INSTRUCTIONS = """Given the question, the evidence and the reference answer, judge the proposed option:
- `also_correct`: the option could also be considered a correct answer to the question as worded.
  Read the whole evidence: an item listed alongside the reference answer (another package in the
  same install command, another member of the same list) is also correct.
- `plausible`: a reader without the source could believe it (not absurd, not a placeholder).
- `same_type`: it has the same semantic type and form as the reference answer (a year for a year,
  an organisation for an organisation, the same unit for a quantity)."""

OPTIONS_ONLY_SYSTEM = "You take multiple-choice tests."
OPTIONS_ONLY_INSTRUCTIONS = """You are shown only the answer options of a multiple-choice question, not the question
itself. Guess which option is most likely the correct one, using any cue you can find (length,
specificity, wording, odd one out). Answer with the option letter only."""


class MultipleChoiceConverter:
    def __init__(self, ctx: GenerationContext, n_options: int = 4, *, probe: LLM | None = None):
        self.ctx = ctx
        self.n_options = n_options
        self.probe = probe  # options-only giveaway probe; None disables it
        self.last_failure: str | None = None

    # ------------------------------------------------------------------ pools
    def _in_document(self, cand: Candidate) -> list[str]:
        """Values/entities in the same *role* as the answer: same predicate, same type and unit."""
        store = self.ctx.store
        out: list[str] = []
        for f in store.many(cand.source_fact_ids):
            obj_role = answers_match(cand.answer, f.object.display()) or (
                f.object.is_numeric and answers_match(cand.answer, fmt_number(f.object.value))
            )
            subj_role = answers_match(cand.answer, f.subject)
            for g in store.facts_for_predicate(f.predicate):
                if obj_role:
                    if g.object.is_numeric != f.object.is_numeric:
                        continue
                    if f.object.is_numeric:
                        if (g.object.unit or "").lower() != (f.object.unit or "").lower():
                            continue
                        if not _close_magnitude(f.object.value, g.object.value, f.object.unit):
                            continue
                    out.append(g.object.display())
                elif subj_role:
                    out.append(g.subject)
        return self._clean(cand, out)

    def _listed_with_answer(self, cand: Candidate) -> list[str]:
        """Items the source states in the same relation as the answer: other items of the answer's
        own list, and objects of sibling facts (same subject, related predicate). Facts about
        *other* subjects stay usable: their values are the best in-document distractors."""
        store = self.ctx.store
        out: list[str] = []
        for f in store.many(cand.source_fact_ids):
            stems = _predicate_stems(f.predicate)
            siblings = [
                g
                for g in store.facts_for_subject(f.subject)
                if g.id == f.id
                or (norm(g.subject) == norm(f.subject) and stems & _predicate_stems(g.predicate))
            ]
            for g in siblings:
                out.extend(_list_items(g.object.display()))
        return out

    def _misconceptions(self, cand: Candidate, closed_book_answers: list[str]) -> list[str]:
        out = [
            a
            for a in closed_book_answers
            if norm(a) not in _NON_ANSWERS
            and len(a.split()) <= max(self.ctx.max_answer_words, len(cand.answer.split()) + 2)
        ]
        return self._clean(cand, out)

    def _near_miss(self, cand: Candidate) -> list[str]:
        m = _NUM.match(cand.answer)
        if not m:
            return []
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            return []
        unit = m.group(2)
        decimals = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
        rng = self.ctx.rng
        vals: list[float] = []
        if is_year_like(value, unit or None):
            vals = [value + d for d in rng.sample([-5, -3, -2, -1, 1, 2, 3, 5], 5)]
        elif float(value).is_integer() and abs(value) < 20:
            vals = [v for v in (value - 2, value - 1, value + 1, value + 2) if v >= 0]
        else:
            vals = [
                perturb_number(value, rng, text=m.group(1), unit=unit or None) for _ in range(4)
            ]
            if value:
                vals.append(value * 10 if abs(value) < 1 else value / 10)  # factor-of-ten slip
        out = []
        for v in vals:
            s = f"{v:.{decimals}f}" if decimals else fmt_number(round(v))
            out.append(f"{s} {unit}".strip())
        return self._clean(cand, out)

    def _clean(self, cand: Candidate, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        answers = cand.all_answers()
        for v in values:
            v = v.strip()
            if not v or norm(v) in seen or any(answers_match(v, a) for a in answers):
                continue
            seen.add(norm(v))
            out.append(v)
        return out

    # --------------------------------------------------------------- convert
    async def convert(
        self,
        cand: Candidate,
        evidence: list[str],
        closed_book_answers: list[str] | None = None,
    ) -> Candidate | None:
        self.last_failure = None
        k = self.n_options - 1
        pools = {
            "in_document": self._in_document(cand)[:10],
            "misconception": self._misconceptions(cand, closed_book_answers or [])[:5],
            "near_miss": self._near_miss(cand)[:6],
        }
        payload = {
            "question": cand.question,
            "correct_answer": cand.answer,
            "evidence": evidence,
            "pools": pools,
        }
        try:
            resp = await self.ctx.llm.generate(
                build_messages(DISTRACTOR_SYSTEM, DISTRACTOR_INSTRUCTIONS.format(k=k + 3), payload),
                Distractors,
                task="distractors",
            )
            proposals = list(resp.parsed.distractors)  # type: ignore[union-attr]
        except LLMError as exc:
            log.warning("distractor generation failed: %s", exc)
            proposals = []
        # pool items the LLM did not use remain available as backups, with their own kind
        for kind, values in pools.items():
            proposals.extend(DistractorProposal(text=v, kind=kind) for v in values)
        candidates: list[DistractorProposal] = []
        seen: set[str] = set()
        answers = cand.all_answers()
        for p in proposals:
            t = p.text.strip()
            if not t or norm(t) in seen or any(answers_match(t, a) for a in answers):
                continue
            if _PLACEHOLDER.search(t):
                continue
            seen.add(norm(t))
            kind = p.kind if p.kind in _KIND_RANK else "other"
            candidates.append(DistractorProposal(text=t, kind=kind))
        # most deceptive kinds first, then closest in form to the correct answer
        candidates.sort(key=lambda p: (_KIND_RANK[p.kind], _form_distance(cand.answer, p.text)))
        # options the document lists alongside the answer are also correct ("pip install wheel"
        # next to "pip install numpy"); an LLM audit misses these surprisingly often
        listed = self._listed_with_answer(cand)
        candidates = [p for p in candidates if not any(answers_match(p.text, x) for x in listed)]
        candidates = candidates[: k + 5]  # bound the number of audit calls
        checks = await asyncio.gather(*(self._check(cand, p.text, evidence) for p in candidates))
        valid = [p for p, ok in zip(candidates, checks, strict=True) if ok]
        chosen = _pick(cand.answer, valid, k)
        if len(chosen) < k:
            self.last_failure = "too_few_distractors"
            return None
        if _stands_out(cand.answer, [p.text for p in chosen]):
            self.last_failure = "answer_stands_out"
            return None
        options = [cand.answer, *(p.text for p in chosen)]
        self.ctx.rng.shuffle(options)
        hits = await self._options_only_hits(options, cand.answer)
        if hits is not None and hits >= 2:
            self.last_failure = "options_only_guessable"
            return None
        letters = "ABCDEFGH"
        correct_idx = options.index(cand.answer)
        return cand.model_copy(
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
                    "distractors": [p.text for p in chosen],
                    "distractor_kinds": {p.text: p.kind for p in chosen},
                    "options_only_hits": hits,
                },
            }
        )

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
        return (not v.also_correct) and v.plausible and v.same_type

    async def _options_only_hits(self, options: list[str], correct: str) -> int | None:
        """How many of two shuffled options-only probes pick the correct answer (None = skipped)."""
        if self.probe is None:
            return None
        letters = "ABCDEFGH"
        orders = [list(options), list(reversed(options))]
        hits = 0
        for trial, order in enumerate(orders):
            payload = {"options": [f"{letters[i]}) {o}" for i, o in enumerate(order)]}
            try:
                resp = await self.probe.generate(
                    build_messages(OPTIONS_ONLY_SYSTEM, OPTIONS_ONLY_INSTRUCTIONS, payload),
                    OptionsOnlyPick,
                    task="options_only",
                    trial=trial,
                )
            except LLMError:
                return None
            pick = resp.parsed.answer.strip()[:1].upper()  # type: ignore[union-attr]
            if pick in letters[: len(order)] and order[letters.index(pick)] == correct:
                hits += 1
        return hits


_STOP_PRED = {"is", "has", "have", "the", "of", "to", "for", "with", "by", "in", "on", "a", "an"}


def _predicate_stems(predicate: str) -> set[str]:
    """Crude stems of a predicate's words: "installs_packages" → {"insta", "packa"}."""
    words = re.split(r"[_\s]+", predicate.lower())
    return {w[:5] for w in words if len(w) >= 3 and w not in _STOP_PRED}


def _list_items(text: str) -> list[str]:
    """Items of an enumerated object ("pip, wheel, setuptools and cython")."""
    parts = re.split(r",|;|\band\b|\bor\b|\s\+\s", text)
    return [p.strip(" .") for p in parts if p.strip(" .")]


_PLACEHOLDER = re.compile(
    r"\b(?i:institution|method|model|group|company|person|dataset|option|lab) [A-Z]\b"
    r"|\b(?i:(?:all|none) of the above)\b"
)


def _close_magnitude(a: float | None, b: float | None, unit: str | None) -> bool:
    if a is None or b is None:
        return False
    if is_year_like(a, unit) or is_year_like(b, unit):
        return abs(a - b) <= 40
    if a == 0 or b == 0:
        return True
    r = abs(a / b)
    return 0.2 <= r <= 5.0


def _form_distance(answer: str, option: str) -> float:
    """Rough difference in surface form: word count, character length, numeric-ness."""
    wa, wo = len(answer.split()), len(option.split())
    la, lo = max(1, len(answer)), max(1, len(option))
    num_a, num_o = bool(_NUM.match(answer)), bool(_NUM.match(option))
    return abs(wa - wo) + abs(la - lo) / la + (2.0 if num_a != num_o else 0.0)


def _pick(answer: str, valid: list[DistractorProposal], k: int) -> list[DistractorProposal]:
    """Take the k best distractors, but no more than k-1 of the same kind when there is a choice,
    so the option set does not share one tell (e.g. all near-miss numbers around the answer)."""
    chosen: list[DistractorProposal] = []
    for p in valid:
        if len(chosen) == k:
            break
        same = sum(1 for c in chosen if c.kind == p.kind)
        others_left = any(q.kind != p.kind for q in valid if q not in chosen and q is not p)
        if k > 1 and same >= k - 1 and others_left:
            continue
        chosen.append(p)
    for p in valid:  # fill up if diversity could not be satisfied
        if len(chosen) == k:
            break
        if p not in chosen:
            chosen.append(p)
    return chosen


def _stands_out(answer: str, distractors: list[str]) -> bool:
    """Classic test-wise cues: the correct option is by far the longest/most qualified one, or
    the only one with (or without) a number, unit or parenthetical."""
    if not distractors:
        return False
    la = len(answer)
    longest = max(len(d) for d in distractors)
    shortest = min(len(d) for d in distractors)
    if la > 1.6 * longest and la - longest >= 8:
        return True
    if la < 0.5 * shortest and shortest - la >= 8:
        return True
    is_num = bool(_NUM.match(answer))
    if all(bool(_NUM.match(d)) != is_num for d in distractors):
        return True
    has_paren = "(" in answer
    return has_paren and not any("(" in d for d in distractors)
