"""Comparison questions over numeric facts sharing a predicate (design §10.3)."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from trivium.facts.models import Fact
from trivium.facts.store import norm
from trivium.generation.base import Candidate, Generator

INSTRUCTIONS = """The facts report the same quantity for several different subjects. Ask which subject has the
{extreme} value of that quantity. Mention the quantity and any shared conditions explicitly; do not
mention the numeric values themselves. The `answer` is the subject name given in `required_answer`."""

LOWER_IS_BETTER = ("error", "mae", "rmse", "loss", "cost", "time", "latency", "deviation")


def comparable_groups(facts: list[Fact], *, cross_document: bool | None = None) -> list[list[Fact]]:
    """Group numeric facts by (predicate, unit, qualifiers) with ≥2 distinct subjects and values."""
    groups: dict[tuple, list[Fact]] = defaultdict(list)
    for f in facts:
        if not f.object.is_numeric:
            continue
        key = (f.predicate, (f.object.unit or "").lower(), tuple(sorted(f.qualifiers.items())))
        groups[key].append(f)
    out: list[list[Fact]] = []
    for group in groups.values():
        by_subject: dict[str, Fact] = {}
        for f in group:
            by_subject.setdefault(norm(f.subject), f)
        members = list(by_subject.values())
        if len(members) < 2 or len({f.object.value for f in members}) < 2:
            continue
        docs = {f.document_id for f in members}
        if cross_document is True and len(docs) < 2:
            continue
        if cross_document is False and len(docs) > 1:
            members = [f for f in members if f.document_id == members[0].document_id]
            if len(members) < 2 or len({f.object.value for f in members}) < 2:
                continue
        out.append(members)
    return out


class ComparisonGenerator(Generator):
    name = "comparison"
    reasoning_depth = 2
    cross_document: bool | None = None

    def groups(self) -> list[list[Fact]]:
        return comparable_groups(self.ctx.store.all(), cross_document=self.cross_document)

    def available(self) -> bool:
        return bool(self.groups())

    async def propose(self, n: int) -> list[Candidate]:
        groups = self.groups()
        if not groups:
            return []
        weights = [sum(self.ctx.fact_weight(f) for f in g) for g in groups]
        picks = [self.ctx.rng.choices(groups, weights=weights, k=1)[0] for _ in range(n)]
        results = await asyncio.gather(*(self._one(g) for g in picks))
        return [c for c in results if c is not None]

    async def _one(self, group: list[Fact]) -> Candidate | None:
        rng = self.ctx.rng
        members = rng.sample(group, k=min(len(group), rng.choice([2, 3, 4])))
        if len({f.object.value for f in members}) < 2:
            return None
        extreme = rng.choice(["lowest", "highest"])
        key = min if extreme == "lowest" else max
        winner = key(members, key=lambda f: f.object.value)  # type: ignore[arg-type]
        # ties on the extreme make the answer ambiguous
        if sum(1 for f in members if f.object.value == winner.object.value) > 1:
            return None
        gq = await self.ask_llm(
            members,
            INSTRUCTIONS.format(extreme=extreme),
            fixed_answer=winner.subject,
            payload_extra={"comparison": extreme, "quantity": winner.predicate},
        )
        if gq is None:
            return None
        cand = self.ctx.make_candidate(
            gq,
            members,
            self.name,
            reasoning_depth=2,
            derivation={
                "op": f"arg{'min' if extreme == 'lowest' else 'max'}",
                "values": {f.subject: f.object.display() for f in members},
            },
        )
        return cand
