"""Retrieval difficulty: rank of the gold chunk when retrieving over all chunks (design §17)."""

from __future__ import annotations

import numpy as np

from trivium.ingestion.models import Chunk
from trivium.models.embeddings import Embedder


class RetrievalProbe:
    def __init__(self, embedder: Embedder, chunks: list[Chunk], ranks: list[int]):
        self.embedder = embedder
        self.chunks = chunks
        self.ranks = sorted(ranks) or [1, 3, 5, 10]
        self._ids = [c.id for c in chunks]
        self._vecs: np.ndarray | None = None

    async def prepare(self) -> None:
        if self._vecs is None:
            self._vecs = await self.embedder.embed(
                [f"{c.section_title}\n{c.text}" for c in self.chunks]
            )

    async def gold_rank(self, question: str, gold_chunk_ids: list[str]) -> int | None:
        await self.prepare()
        if self._vecs is None or not gold_chunk_ids:
            return None
        q = (await self.embedder.embed([question]))[0]
        sims = self._vecs @ q
        order = np.argsort(-sims)
        gold = set(gold_chunk_ids)
        for rank, idx in enumerate(order, start=1):
            if self._ids[idx] in gold:
                return rank
        return None

    def label(self, rank: int | None) -> str:
        if rank is None:
            return "unknown"
        if rank <= self.ranks[0]:
            return "easy"
        if rank <= self.ranks[min(1, len(self.ranks) - 1)]:
            return "medium"
        if rank <= self.ranks[-1]:
            return "hard"
        return "very_hard"
