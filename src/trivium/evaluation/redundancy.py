"""Redundancy against accepted questions: embedding similarity + fact overlap (design §18)."""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel

from trivium.generation.base import Candidate
from trivium.models.embeddings import Embedder
from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)


class DuplicateVerdict(BaseModel):
    duplicate: bool
    explanation: str = ""


DUP_SYSTEM = "You detect duplicate quiz questions."
DUP_INSTRUCTIONS = """Two questions are duplicates if a person who knows the answer to one necessarily knows the
answer to the other (same fact asked differently). Different facts about the same entity are NOT duplicates."""


class RedundancyIndex:
    def __init__(self, embedder: Embedder, *, gray_zone: tuple[float, float] = (0.75, 0.9)):
        self.embedder = embedder
        self.gray_zone = gray_zone
        self._vecs: np.ndarray | None = None
        self._cands: list[Candidate] = []

    def __len__(self) -> int:
        return len(self._cands)

    async def add(self, cand: Candidate) -> None:
        v = await self.embedder.embed([self._text(cand)])
        self._vecs = v if self._vecs is None else np.vstack([self._vecs, v])
        self._cands.append(cand)

    @staticmethod
    def _text(cand: Candidate) -> str:
        return f"{cand.question} {cand.answer}"

    async def assess(
        self, cand: Candidate, judge: LLM | None = None
    ) -> tuple[float, float, Candidate | None]:
        """Return (max_similarity, fact_overlap, most_similar_accepted)."""
        if not self._cands or self._vecs is None:
            return 0.0, 0.0, None
        v = await self.embedder.embed([self._text(cand)])
        sims = (self._vecs @ v[0]).astype(float)
        idx = int(np.argmax(sims))
        max_sim = float(sims[idx])
        nearest = self._cands[idx]
        cand_facts = set(cand.source_fact_ids)
        overlap = 0.0
        if cand_facts:
            overlap = max(
                len(cand_facts & set(c.source_fact_ids)) / len(cand_facts) for c in self._cands
            )
        lo, hi = self.gray_zone
        if judge is not None and lo <= max_sim < hi:
            payload = {
                "question_a": cand.question,
                "answer_a": cand.answer,
                "question_b": nearest.question,
                "answer_b": nearest.answer,
            }
            try:
                resp = await judge.generate(
                    build_messages(DUP_SYSTEM, DUP_INSTRUCTIONS, payload),
                    DuplicateVerdict,
                    task="judge_duplicate",
                )
                if resp.parsed.duplicate:  # type: ignore[union-attr]
                    max_sim = max(max_sim, hi)
            except LLMError as exc:
                log.warning("duplicate judge failed: %s", exc)
        return max_sim, overlap, nearest
