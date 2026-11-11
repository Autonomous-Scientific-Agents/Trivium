"""True/False statements built from validated facts (design §10.7).

A good false statement is a *minimal edit* of a true one that the evidence clearly contradicts,
and the edit should be the kind a careless reader falls for. Besides changing the object
(numeric perturbation, sibling-object swap), this generator can:

* ``subject_swap`` – credit the value to the competing subject ("XTB achieved an MAE of 0.12 eV"
  when 0.12 eV was MACE's);
* ``qualifier_swap`` – keep subject and quantity, but assert the value from another condition;
* ``comparison_flip`` – reverse the direction of a comparison between two subjects;
* ``llm_edit`` – let the model make one targeted edit (direction, ordinal, quantifier, cause,
  common misconception).

True statements are *paraphrased* (so surface matching against the source does not reveal them),
while false ones stay close to the source wording with a single change. Every statement is
verified by entailment: true ones must be ``entailed``, false ones ``contradicted`` (not merely
absent).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from trivium.config import QuestionType
from trivium.evaluation.entailment import check_statement
from trivium.facts.models import Fact, FactObject, fmt_number
from trivium.facts.store import norm
from trivium.generation.base import Candidate, GeneratedQuestion, Generator
from trivium.models.llm import LLMError
from trivium.prompts import build_messages


class Statement(BaseModel):
    statement: str


class FalseStatement(BaseModel):
    statement: str
    edit: str = ""  # what was changed, e.g. "higher → lower"


STATEMENT_SYSTEM = "You rewrite structured claims as fluent declarative sentences."
STATEMENT_INSTRUCTIONS = """Write the claim as one clear, self-contained declarative sentence of at most {max_words} words.
- A `claim` of type "fact" asserts `object` for `subject` and `predicate` (the object may have been
  deliberately altered: assert it exactly as given, without hedging). A `claim` of type "comparison"
  asserts that `subject` has a `relation` (higher/lower) `quantity` than `other_subject`.
- Include the qualifiers as conditions. Do not add information that is not in the claim.
- Name things by their real names; never write "the paper", "the authors", "Table 2", "Run 3".
- `style` = "paraphrase": reword it; do not copy the phrasing of `evidence` (keep names and numbers
  exact). `style` = "close": follow the wording of `evidence` as closely as the claim allows."""

FALSIFY_SYSTEM = "You write subtly false quiz statements."
FALSIFY_INSTRUCTIONS = """Write one declarative sentence (at most {max_words} words) that a reader of the evidence would
recognise as FALSE, made by a single targeted edit of the fact: reverse a direction (increase ↔
decrease, higher ↔ lower, before ↔ after), change an ordinal or quantifier (first → second, all →
most), swap cause and effect, attribute it to a closely related entity, or state the common
misconception instead of the finding. Keep everything else as in the evidence. The evidence must
clearly CONTRADICT the sentence (not merely fail to mention it). Name things by their real names.
Describe the change in `edit`."""

# relative weights of the ways to falsify, renormalised over the ones applicable to a fact
_FALSE_WEIGHTS = {
    "llm_edit": 0.25,
    "subject_swap": 0.2,
    "qualifier_swap": 0.2,
    "comparison_flip": 0.15,
    "numeric_perturbation": 0.1,
    "sibling_object_swap": 0.1,
}


def _decimals(text: str) -> int:
    m = re.search(r"\d+\.(\d+)", text or "")
    return len(m.group(1)) if m else 0


def is_year_like(value: float, unit: str | None) -> bool:
    return float(value).is_integer() and 1000 <= value <= 2999 and not unit


def perturb_number(value: float, rng, *, text: str = "", unit: str | None = None) -> float:
    """Return a wrong-but-plausible alternative to ``value``.

    Years are shifted by a few years; other quantities are scaled by 0.6–1.6 and rounded to the
    precision of the original so the false statement does not look obviously fabricated.
    """
    if is_year_like(value, unit):
        delta = rng.choice([-12, -7, -5, -3, -2, 2, 3, 5, 7, 12])
        return float(int(value) + delta)
    if value == 0:
        return rng.choice([1.0, -1.0, 0.5])
    factor = rng.choice([0.6, 0.75, 0.85, 1.15, 1.3, 1.6])
    if unit and unit.strip().lower() in ("%", "percent", "percentage") and value * factor > 100:
        factor = rng.choice([0.6, 0.75, 0.85])  # keep percentages within [0, 100]
    new = value * factor
    decimals = _decimals(text)
    if decimals:
        new = round(new, decimals)
    elif float(value).is_integer():
        # keep integer-ness and roughly the same number of significant digits
        mag = 10 ** max(0, len(str(int(abs(value)))) - 2)
        new = float(round(new / mag) * mag)
        if new == value:  # rounding collapsed the change: step one unit instead
            new = value + (mag if factor > 1 else -mag)
            if new <= 0:
                new = value + mag
    else:
        new = float(f"{new:.3g}")
    return new if new != value else value * 1.5


@dataclass
class Claim:
    kind: str
    truth: bool
    facts: list[Fact]  # first = the fact the statement is about
    asserted: str
    payload: dict[str, Any] = field(default_factory=dict)  # for the statement writer

    @property
    def evidence(self) -> list[str]:
        return list(dict.fromkeys(f.evidence for f in self.facts))


class TrueFalseGenerator(Generator):
    name = "true_false"
    question_type = QuestionType.true_false

    _partners: dict[str, list[Fact]] | None = None

    async def propose(self, n: int) -> list[Candidate]:
        facts = self.ctx.sample_facts(n)
        results = await asyncio.gather(*(self._one(f) for f in facts))
        return [c for c in results if c is not None]

    # ------------------------------------------------------------ partners
    def comparison_partners(self, fact: Fact) -> list[Fact]:
        """Numeric facts reporting the same quantity (predicate, unit, qualifiers) for another
        subject with a different value: the material for subject swaps and comparisons."""
        if self._partners is None:
            from trivium.generation.comparison import comparable_groups

            self._partners = {}
            for group in comparable_groups(self.ctx.store.all()):
                for f in group:
                    self._partners[f.id] = [
                        g for g in group if g.id != f.id and g.object.value != f.object.value
                    ]
        return self._partners.get(fact.id, [])

    def _subject_swap_partners(self, fact: Fact) -> list[Fact]:
        if fact.object.is_numeric:
            return self.comparison_partners(fact)
        return [
            g
            for g in self.ctx.store.facts_for_predicate(fact.predicate)
            if norm(g.subject) != norm(fact.subject)
            and not g.object.is_numeric
            and norm(g.object.text) != norm(fact.object.text)
        ]

    def _qualifier_partners(self, fact: Fact) -> list[Fact]:
        return [
            g
            for g in self.ctx.store.facts_for_subject(fact.subject)
            if g.id != fact.id
            and g.predicate == fact.predicate
            and g.qualifiers != fact.qualifiers
            and norm(g.object.display()) != norm(fact.object.display())
        ]

    # --------------------------------------------------------------- claims
    @staticmethod
    def _fact_payload(subject: str, fact: Fact, obj: str, qualifiers: dict) -> dict[str, Any]:
        return {
            "claim": "fact",
            "subject": subject,
            "predicate": fact.predicate,
            "object": obj,
            "qualifiers": qualifiers,
        }

    @staticmethod
    def _comparison_payload(a: Fact, b: Fact, relation: str) -> dict[str, Any]:
        return {
            "claim": "comparison",
            "subject": a.subject,
            "relation": relation,
            "quantity": a.predicate,
            "other_subject": b.subject,
            "qualifiers": a.qualifiers,
        }

    def _true_claim(self, fact: Fact) -> Claim:
        rng = self.ctx.rng
        partners = self.comparison_partners(fact)
        if partners and rng.random() < 0.3:
            b = rng.choice(partners)
            rel = "higher" if fact.object.value > b.object.value else "lower"  # type: ignore[operator]
            return Claim(
                "comparison_true",
                True,
                [fact, b],
                f"{rel} than {b.subject}",
                self._comparison_payload(fact, b, rel),
            )
        obj = fact.object.display()
        return Claim(
            "verbatim",
            True,
            [fact],
            obj,
            self._fact_payload(fact.subject, fact, obj, fact.qualifiers),
        )

    def _false_claim(self, fact: Fact, kind: str | None = None) -> Claim | None:
        """A false claim derived from ``fact``; ``kind`` forces one falsification (None if it does
        not apply to this fact), otherwise one is sampled from the applicable ones."""
        rng = self.ctx.rng
        options: dict[str, Any] = {"llm_edit": None}
        if fact.object.is_numeric:
            options["numeric_perturbation"] = None
        else:
            siblings = [
                f
                for f in self.ctx.store.facts_for_predicate(fact.predicate)
                if norm(f.object.text) != norm(fact.object.text) and not f.object.is_numeric
            ]
            if siblings:
                options["sibling_object_swap"] = siblings
        if swap := self._subject_swap_partners(fact):
            options["subject_swap"] = swap
        if qual := self._qualifier_partners(fact):
            options["qualifier_swap"] = qual
        if comp := self.comparison_partners(fact):
            options["comparison_flip"] = comp
        if kind is None:
            kinds = list(options)
            kind = rng.choices(kinds, weights=[_FALSE_WEIGHTS[k] for k in kinds], k=1)[0]
        elif kind not in options:
            return None
        material = options[kind]

        if kind == "llm_edit":
            return Claim("llm_edit", False, [fact], "", {})
        if kind == "numeric_perturbation":
            v = perturb_number(fact.object.value, rng, text=fact.object.text, unit=fact.object.unit)  # type: ignore[arg-type]
            obj = FactObject(text=fmt_number(v), value=v, unit=fact.object.unit).display()
            return Claim(
                kind,
                False,
                [fact],
                obj,
                self._fact_payload(fact.subject, fact, obj, fact.qualifiers),
            )
        if kind == "sibling_object_swap":
            obj = rng.choice(material).object.text
            return Claim(
                kind,
                False,
                [fact],
                obj,
                self._fact_payload(fact.subject, fact, obj, fact.qualifiers),
            )
        if kind == "subject_swap":
            # the statement is about g (the other subject) but asserts fact's value
            g = rng.choice(material)
            obj = fact.object.display()
            return Claim(
                kind, False, [g, fact], obj, self._fact_payload(g.subject, fact, obj, g.qualifiers)
            )
        if kind == "qualifier_swap":
            h = rng.choice(material)  # same subject/quantity, other condition
            obj = h.object.display()
            return Claim(
                kind,
                False,
                [fact, h],
                obj,
                self._fact_payload(fact.subject, fact, obj, fact.qualifiers),
            )
        # comparison_flip
        b = rng.choice(material)
        wrong = "lower" if fact.object.value > b.object.value else "higher"  # type: ignore[operator]
        return Claim(
            kind,
            False,
            [fact, b],
            f"{wrong} than {b.subject}",
            self._comparison_payload(fact, b, wrong),
        )

    # ------------------------------------------------------------ writing
    async def _write(self, claim: Claim) -> str | None:
        max_words = self.ctx.config.generation.max_statement_words
        try:
            if claim.kind == "llm_edit":
                f = claim.facts[0]
                payload = {
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object.display(),
                    "qualifiers": f.qualifiers,
                    "evidence": f.evidence,
                }
                resp = await self.ctx.llm.generate(
                    build_messages(
                        FALSIFY_SYSTEM, FALSIFY_INSTRUCTIONS.format(max_words=max_words), payload
                    ),
                    FalseStatement,
                    task="falsify_statement",
                )
                claim.asserted = resp.parsed.edit or "llm_edit"  # type: ignore[union-attr]
                return resp.parsed.statement.strip()  # type: ignore[union-attr]
            payload = {
                **claim.payload,
                "style": "paraphrase" if claim.truth else "close",
                "evidence": claim.evidence,
            }
            resp = await self.ctx.llm.generate(
                build_messages(
                    STATEMENT_SYSTEM, STATEMENT_INSTRUCTIONS.format(max_words=max_words), payload
                ),
                Statement,
                task="write_statement",
            )
            return resp.parsed.statement.strip()  # type: ignore[union-attr]
        except LLMError:
            return None

    async def _one(self, fact: Fact) -> Candidate | None:
        make_false = self.ctx.rng.random() < 0.5
        claim = self._false_claim(fact) if make_false else self._true_claim(fact)
        if claim is None:
            return None
        return await self._realise(claim)

    async def _realise(self, claim: Claim) -> Candidate | None:
        """Write the statement, enforce the length cap and verify it against the evidence."""
        statement = await self._write(claim)
        if not statement:
            return None
        if len(statement.split()) > self.ctx.config.generation.max_statement_words:
            return None  # long, citation-like statements make poor quiz items
        verdict = await check_statement(self.ctx.llm, statement, claim.evidence)
        expected = "entailed" if claim.truth else "contradicted"
        if verdict.label != expected:
            return None  # false statements must be *contradicted*, not merely absent (design §10.7)
        answer = "True" if claim.truth else "False"
        gq = GeneratedQuestion(
            question=f"True or False: {statement}",
            answer=answer,
            accepted_answers=[answer.lower(), "T" if claim.truth else "F"],
            answer_type="boolean",
        )
        cand = self.ctx.make_candidate(
            gq,
            claim.facts,
            self.name,
            reasoning_depth=2 if len(claim.facts) > 1 else 1,
            derivation={"perturbation": claim.kind, "asserted_object": claim.asserted},
        )
        cand.answer_type = "boolean"
        cand.answer_kind = None
        cand.question_type = QuestionType.true_false
        return cand
