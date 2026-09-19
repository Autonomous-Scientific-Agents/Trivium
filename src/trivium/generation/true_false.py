"""True/False statements built by (optionally) perturbing validated facts (design §10.7)."""

from __future__ import annotations

import asyncio
import re

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


STATEMENT_SYSTEM = "You rewrite structured facts as fluent declarative sentences."
STATEMENT_INSTRUCTIONS = """Rewrite the fact as one clear, self-contained declarative sentence that asserts the object value
given (`object` may have been deliberately altered — assert it as given without hedging). Include the
qualifiers as conditions. Do not add information that is not in the fact."""


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


class TrueFalseGenerator(Generator):
    name = "true_false"
    question_type = QuestionType.true_false

    async def propose(self, n: int) -> list[Candidate]:
        facts = self.ctx.sample_facts(n)
        results = await asyncio.gather(*(self._one(f) for f in facts))
        return [c for c in results if c is not None]

    def _alternative_object(self, fact: Fact) -> tuple[str, str] | None:
        """Return (object_text, source) for a false variant, or None if none available."""
        rng = self.ctx.rng
        if fact.object.is_numeric:
            v = perturb_number(fact.object.value, rng, text=fact.object.text, unit=fact.object.unit)  # type: ignore[arg-type]
            txt = FactObject(text=fmt_number(v), value=v, unit=fact.object.unit).display()
            return txt, "numeric_perturbation"
        siblings = [
            f
            for f in self.ctx.store.facts_for_predicate(fact.predicate)
            if norm(f.object.text) != norm(fact.object.text) and not f.object.is_numeric
        ]
        if siblings:
            return rng.choice(siblings).object.text, "sibling_object_swap"
        return None

    async def _one(self, fact: Fact) -> Candidate | None:
        rng = self.ctx.rng
        make_false = rng.random() < 0.5
        obj_text = fact.object.display()
        source = "verbatim"
        if make_false:
            alt = self._alternative_object(fact)
            if alt is None:
                make_false = False
            else:
                obj_text, source = alt
        payload = {
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": obj_text,
            "qualifiers": fact.qualifiers,
        }
        try:
            resp = await self.ctx.llm.generate(
                build_messages(STATEMENT_SYSTEM, STATEMENT_INSTRUCTIONS, payload),
                Statement,
                task="write_statement",
            )
        except LLMError:
            return None
        statement = resp.parsed.statement.strip()  # type: ignore[union-attr]
        if not statement:
            return None
        verdict = await check_statement(self.ctx.llm, statement, [fact.evidence])
        expected = "contradicted" if make_false else "entailed"
        if verdict.label != expected:
            return None  # false statements must be *contradicted*, not merely absent (design §10.7)
        answer = "False" if make_false else "True"
        gq = GeneratedQuestion(
            question=f"True or False: {statement}",
            answer=answer,
            accepted_answers=[answer.lower(), "T" if answer == "True" else "F"],
            answer_type="boolean",
        )
        cand = self.ctx.make_candidate(
            gq, [fact], self.name, derivation={"perturbation": source, "asserted_object": obj_text}
        )
        cand.answer_type = "boolean"
        cand.question_type = QuestionType.true_false
        return cand
