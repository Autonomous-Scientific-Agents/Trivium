"""Cross-document questions: comparisons or hops whose facts span ≥2 documents (design §10.6)."""

from __future__ import annotations

from trivium.generation.base import Candidate, Generator
from trivium.generation.comparison import ComparisonGenerator
from trivium.generation.multihop import MultiHopGenerator


class _XComparison(ComparisonGenerator):
    cross_document = True


class _XMultiHop(MultiHopGenerator):
    cross_document = True


class CrossDocumentGenerator(Generator):
    name = "cross_document"
    reasoning_depth = 2

    def __init__(self, ctx):
        super().__init__(ctx)
        self._cmp = _XComparison(ctx)
        self._hop = _XMultiHop(ctx)

    def available(self) -> bool:
        return len(self.ctx.store.documents()) >= 2 and (
            self._cmp.available() or self._hop.available()
        )

    async def propose(self, n: int) -> list[Candidate]:
        subs = [g for g in (self._cmp, self._hop) if g.available()]
        if not subs:
            return []
        per = max(1, n // len(subs))
        out: list[Candidate] = []
        for g in subs:
            cands = await g.propose(per)
            for c in cands:
                if len(set(c.source_document_ids)) >= 2:
                    c.derivation = {**(c.derivation or {}), "via": g.name}
                    out.append(c)
        return out[:n]
