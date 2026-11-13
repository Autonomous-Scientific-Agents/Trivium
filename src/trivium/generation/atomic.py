"""Atomic-fact questions (design §10.1)."""

from __future__ import annotations

import asyncio

from trivium.facts.models import fmt_number
from trivium.generation.base import Candidate, Generator, answers_match

INSTRUCTIONS = """Write one question that is answered by the single fact below.
Usually ask for the object (value) given the subject and relation; if the object is a very
common word or a long phrase, ask for the subject instead. Set `answer_type` to "numeric" when the
answer is a number (with unit), otherwise "short_answer". List obvious equivalent forms of the
answer in `accepted_answers` (e.g. "0.12 eV" and "0.12")."""


class AtomicGenerator(Generator):
    name = "atomic"

    async def propose(self, n: int) -> list[Candidate]:
        facts = self.ctx.sample_facts(n)
        results = await asyncio.gather(*(self._one(f) for f in facts))
        return [c for c in results if c is not None]

    async def _one(self, fact) -> Candidate | None:
        gq = await self.ask_llm([fact], INSTRUCTIONS)
        if gq is None:
            return None
        grounded = (
            answers_match(gq.answer, fact.object.display())
            or answers_match(gq.answer, fact.subject)
            or (fact.object.is_numeric and answers_match(gq.answer, fmt_number(fact.object.value)))
        )
        if not grounded:
            return None
        if fact.object.is_numeric and answers_match(gq.answer, fact.object.display()):
            gq.answer_type = "numeric"
            if gq.answer_kind not in ("number", "date"):
                gq.answer_kind = "number"
            for alt in (fmt_number(fact.object.value), fact.object.display(), fact.object.text):
                if alt and alt not in gq.accepted_answers:
                    gq.accepted_answers.append(alt)
        return self.ctx.make_candidate(gq, [fact], self.name)
