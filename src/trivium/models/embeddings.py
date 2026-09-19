"""Embedding backends used for redundancy detection and retrieval-difficulty estimation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from typing import Any

import numpy as np

from trivium.models.cache import ResponseCache, stable_hash

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


class Embedder:
    name: str = "base"
    dim: int = 0

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) L2-normalised float32 array."""
        raise NotImplementedError

    async def embed_one(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class HashingEmbedder(Embedder):
    """Dependency-free hashed word + bigram bag-of-words embedding.

    Good enough for near-duplicate detection and coarse lexical retrieval; swap in
    :class:`OpenAIEmbedder` or :class:`SentenceTransformerEmbedder` for semantic quality.
    """

    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        for f in feats:
            h = int(hashlib.blake2b(f.encode(), digest_size=8).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 63) else -1.0
        return v

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vec(t) for t in texts]))


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        cache: ResponseCache | None = None,
        batch_size: int = 64,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install 'trivium[openai]' to use openai embeddings") from exc
        self.model = model
        self.cache = cache
        self.batch_size = batch_size
        self._client = AsyncOpenAI(
            api_key=os.environ.get(api_key_env) or "none",
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    async def embed(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            key = stable_hash({"emb": self.model, "text": t}) if self.cache else None
            hit = self.cache.get(key) if key else None
            if hit is not None:
                out[i] = np.asarray(json.loads(hit), dtype=np.float32)
            else:
                todo.append(i)
        for start in range(0, len(todo), self.batch_size):
            idx = todo[start : start + self.batch_size]
            resp = await self._client.embeddings.create(
                model=self.model, input=[texts[i] for i in idx]
            )
            for j, d in zip(idx, resp.data, strict=True):
                vec = np.asarray(d.embedding, dtype=np.float32)
                out[j] = vec
                if self.cache:
                    self.cache.set(
                        stable_hash({"emb": self.model, "text": texts[j]}), json.dumps(vec.tolist())
                    )
        arr = np.stack([o for o in out if o is not None]) if out else np.zeros((0, 1))
        self.dim = arr.shape[1]
        return _normalise(arr)


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install 'trivium[embeddings]' for local embeddings") from exc
        self._model = SentenceTransformer(model)
        self.model = model

    async def embed(self, texts: list[str]) -> np.ndarray:
        loop = asyncio.get_running_loop()
        arr = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, convert_to_numpy=True)
        )
        self.dim = arr.shape[1]
        return _normalise(np.asarray(arr, dtype=np.float32))


def build_embedder(
    provider: str, model: str | None = None, *, cache: ResponseCache | None = None, **kw: Any
) -> Embedder:
    provider = (provider or "hashing").lower()
    if provider in ("hashing", "local-hash", "mock"):
        return HashingEmbedder()
    if provider in ("openai", "openai-compatible"):
        return OpenAIEmbedder(model or "text-embedding-3-small", cache=cache, **kw)
    if provider == "argo":
        from trivium.models.argo import ArgoEmbedder

        return ArgoEmbedder(model or "v3small", cache=cache, **kw)
    if provider in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(model or "all-MiniLM-L6-v2")
    raise ValueError(f"unknown embedding provider: {provider!r}")
