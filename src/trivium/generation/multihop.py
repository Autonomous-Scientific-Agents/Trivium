"""Multi-hop questions from two-edge graph paths (design §10.5)."""

from __future__ import annotations

import asyncio

from trivium.facts.store import norm
from trivium.generation.base import Candidate, Generator
from trivium.graph.backend import Edge

INSTRUCTIONS = """Two linked facts form a chain: A —relation1→ B —relation2→ C.
Write ONE question that starts from A and asks for C without naming B (the bridge entity), so that
answering requires combining both facts. Example: "Which institution created the dataset used to
train Model A?" The `answer` must be exactly `required_answer` (C)."""


class MultiHopGenerator(Generator):
    name = "multi_hop"
    reasoning_depth = 2
    cross_document: bool | None = None

    def paths(self) -> list[tuple[Edge, Edge]]:
        paths = self.ctx.graph.two_hop_paths()
        if self.cross_document is True:
            paths = [p for p in paths if p[0].document_id != p[1].document_id]
        return paths

    def available(self) -> bool:
        return bool(self.paths())

    async def propose(self, n: int) -> list[Candidate]:
        paths = self.paths()
        if not paths:
            return []
        store = self.ctx.store
        weights = [
            self.ctx.fact_weight(store.get(a.fact_id)) + self.ctx.fact_weight(store.get(b.fact_id))
            for a, b in paths
        ]
        picks = [self.ctx.rng.choices(paths, weights=weights, k=1)[0] for _ in range(n)]
        results = await asyncio.gather(*(self._one(a, b) for a, b in picks))
        return [c for c in results if c is not None]

    async def _one(self, e1: Edge, e2: Edge) -> Candidate | None:
        f1, f2 = self.ctx.store.get(e1.fact_id), self.ctx.store.get(e2.fact_id)
        answer = f2.object.display()
        bridge = f2.subject
        gq = await self.ask_llm(
            [f1, f2],
            INSTRUCTIONS,
            fixed_answer=answer,
            payload_extra={
                "A": f1.subject,
                "relation1": f1.predicate,
                "B": bridge,
                "relation2": f2.predicate,
                "C": answer,
            },
        )
        if gq is None:
            return None
        q = norm(gq.question)
        if norm(bridge) in q or norm(answer) in q:
            return None  # bridge or answer leaked → not really multi-hop
        return self.ctx.make_candidate(
            gq,
            [f1, f2],
            self.name,
            reasoning_depth=2,
            graph_path=[f1.subject, f1.predicate, bridge, f2.predicate, answer],
        )
