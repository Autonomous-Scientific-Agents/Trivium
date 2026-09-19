"""Derived numerical questions: differences and ratios computed from two facts (design §10.4)."""

from __future__ import annotations

import asyncio
import math

from trivium.facts.models import Fact, fmt_number
from trivium.generation.base import Candidate, Generator
from trivium.generation.comparison import comparable_groups

INSTRUCTIONS = """Two facts report the same quantity under different subjects or conditions. Ask a question whose
answer is the {op} between them, exactly as given in `required_answer`. Name both subjects/conditions
and the quantity; do not state either value. For a ratio, say which is divided by which and ask for
the value rounded to two decimals."""


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    digits = max(0, 3 - int(math.floor(math.log10(abs(x)))))
    s = f"{round(x, digits):.{digits}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


class DerivedGenerator(Generator):
    name = "derived"
    reasoning_depth = 2

    def pairs(self) -> list[tuple[Fact, Fact]]:
        out = []
        for g in comparable_groups(self.ctx.store.all()):
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    out.append((g[i], g[j]))
        # same subject, same predicate, different qualifiers (e.g. initial vs final)
        by_key: dict[tuple[str, str, str], list[Fact]] = {}
        for f in self.ctx.store.numeric_facts():
            by_key.setdefault(
                (f.subject.lower(), f.predicate, (f.object.unit or "").lower()), []
            ).append(f)
        for group in by_key.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if (
                        group[i].qualifiers != group[j].qualifiers
                        and group[i].object.value != group[j].object.value
                    ):
                        out.append((group[i], group[j]))
        return out

    def available(self) -> bool:
        return bool(self.pairs())

    async def propose(self, n: int) -> list[Candidate]:
        pairs = self.pairs()
        if not pairs:
            return []
        weights = [self.ctx.fact_weight(a) + self.ctx.fact_weight(b) for a, b in pairs]
        picks = [self.ctx.rng.choices(pairs, weights=weights, k=1)[0] for _ in range(n)]
        results = await asyncio.gather(*(self._one(a, b) for a, b in picks))
        return [c for c in results if c is not None]

    async def _one(self, a: Fact, b: Fact) -> Candidate | None:
        va, vb = a.object.value, b.object.value
        assert va is not None and vb is not None
        op = self.ctx.rng.choice(["difference", "ratio"]) if vb != 0 and va != 0 else "difference"
        if op == "difference":
            hi, lo = (a, b) if va >= vb else (b, a)
            value = hi.object.value - lo.object.value  # type: ignore[operator]
            unit = a.object.unit or ""
            answer = f"{_fmt(value)} {unit}".strip()
            accepted = [_fmt(value), fmt_number(value), f"{fmt_number(value)} {unit}".strip()]
            derivation = {
                "op": "difference",
                "minuend": hi.id,
                "subtrahend": lo.id,
                "value": value,
                "unit": unit,
            }
            extra = {
                "operation": f"{hi.subject} minus {lo.subject}"
                if hi.subject != lo.subject
                else f"{hi.qualifiers} minus {lo.qualifiers}"
            }
        else:
            value = round(va / vb, 2)
            if not (0.05 < value < 20):
                return None
            answer = _fmt(value)
            accepted = [f"{value:.2f}", fmt_number(value)]
            derivation = {"op": "ratio", "numerator": a.id, "denominator": b.id, "value": value}
            extra = {
                "operation": f"{a.subject} divided by {b.subject}"
                if a.subject != b.subject
                else f"{a.qualifiers} divided by {b.qualifiers}"
            }
        gq = await self.ask_llm(
            [a, b], INSTRUCTIONS.format(op=op), fixed_answer=answer, payload_extra=extra
        )
        if gq is None:
            return None
        gq.answer_type = "numeric"
        gq.accepted_answers = list(dict.fromkeys([*gq.accepted_answers, *accepted]))
        return self.ctx.make_candidate(
            gq, [a, b], self.name, reasoning_depth=2, derivation=derivation
        )
