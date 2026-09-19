"""Relationship questions from an entity and one of its graph edges (design §10.2)."""

from __future__ import annotations

import asyncio

from trivium.facts.models import fmt_number
from trivium.facts.store import norm
from trivium.generation.base import Candidate, Generator

INSTRUCTIONS = """The target fact links an entity to one of its relations. The entity has several other
relations (listed as context) — use them only to make the question unambiguous, never as the answer.
Ask for the object of the target relation (e.g. "Which catalyst was used in Experiment 4?")."""


class RelationshipGenerator(Generator):
    name = "relationship"

    def _hub_nodes(self) -> list[str]:
        return [n for n in self.ctx.graph.nodes() if len(self.ctx.graph.out_edges(n)) >= 2]

    def available(self) -> bool:
        return bool(self._hub_nodes())

    async def propose(self, n: int) -> list[Candidate]:
        hubs = self._hub_nodes()
        if not hubs:
            return []
        picks = []
        for _ in range(n):
            node = self.ctx.rng.choice(hubs)
            edges = self.ctx.graph.out_edges(node)
            facts = self.ctx.store.many(e.fact_id for e in edges)
            if not facts:
                continue
            target = self.ctx.sample_facts(1, facts, distinct_chunks=False)
            if not target:
                continue
            picks.append((target[0], [f for f in facts if f.id != target[0].id][:6]))
        results = await asyncio.gather(*(self._one(t, ctx_facts) for t, ctx_facts in picks))
        return [c for c in results if c is not None]

    async def _one(self, target, context_facts) -> Candidate | None:
        gq = await self.ask_llm(
            [target],
            INSTRUCTIONS,
            fixed_answer=target.object.display(),
            payload_extra={
                "target_fact_id": target.id,
                "context_relations": [f.statement() for f in context_facts],
            },
        )
        if gq is None:
            return None
        if norm(target.object.text) in norm(gq.question):
            return None  # answer leaked into the question
        if target.object.is_numeric:
            gq.answer_type = "numeric"
            gq.accepted_answers = list(
                {*gq.accepted_answers, fmt_number(target.object.value), target.object.text}
            )
        return self.ctx.make_candidate(gq, [target], self.name)
