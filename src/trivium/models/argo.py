"""Argonne Argo Gateway API provider (chat + embeddings).

Argo exposes OpenAI, Anthropic and Google models behind one internal gateway. This module
talks to the *native* endpoints (available in prod), which differ from the OpenAI schema:

* chat:  ``POST {base}/argoapi/api/v1/resource/chat/``  with ``{"user", "model", "messages",
  "temperature", "top_p", "max_tokens"}`` → ``{"response": "<text>"}``
* embed: ``POST {base}/argoapi/api/v1/resource/embed/`` with ``{"user", "model", "prompt": [...]}``
  → ``{"embedding": [[...], ...]}``

Every call must carry your ANL domain username in ``user`` (read from ``$ARGO_USER`` by default).
The machine must be on the Argonne network or VPN. Reasoning models (o-series, gpt5*) reject
``temperature``/``top_p`` and use ``max_completion_tokens``; this is handled automatically.

Argo also offers an OpenAI-compatible shim (dev only, ``https://apps-dev.inside.anl.gov/argoapi/v1``
with ``api_key=<ANL username>``); that route works through the ``openai`` provider with
``base_url`` and ``api_key_env: ARGO_USER`` and needs nothing from this module.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import numpy as np

from trivium.models.cache import ResponseCache, stable_hash
from trivium.models.embeddings import Embedder, _normalise
from trivium.models.llm import LLM

ARGO_BASES = {
    "prod": "https://apps.inside.anl.gov",
    "test": "https://apps-test.inside.anl.gov",
    "dev": "https://apps-dev.inside.anl.gov",
}
CHAT_PATH = "/argoapi/api/v1/resource/chat/"
EMBED_PATH = "/argoapi/api/v1/resource/embed/"

# Models that take max_completion_tokens and ignore/reject sampling parameters.
_REASONING = re.compile(r"^(gpto\d|gpt5|o\d)", re.I)


class ArgoConfigError(RuntimeError):
    pass


def resolve_user(user: str | None, user_env: str) -> str:
    u = user or os.environ.get(user_env)
    if not u:
        raise ArgoConfigError(
            f"Argo requires your ANL domain username: set ${user_env} or `user:` in the model spec"
        )
    return u


def resolve_base(env: str | None, base_url: str | None) -> str:
    if base_url:
        return base_url.rstrip("/")
    return ARGO_BASES[(env or "prod").lower()]


def _client_kwargs(timeout: float) -> dict[str, Any]:
    return {"timeout": timeout, "trust_env": True}


def build_chat_payload(
    *, user: str, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {"user": user, "model": model, "messages": list(messages)}
    if _REASONING.match(model):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload.update({"temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens})
    return payload


def parse_chat_response(data: Any) -> str:
    """Argo returns ``{"response": text}``; be tolerant of OpenAI-shaped bodies too."""
    if isinstance(data, dict):
        if isinstance(data.get("response"), str):
            return data["response"]
        if isinstance(data.get("choices"), list) and data["choices"]:
            msg = data["choices"][0].get("message", {})
            return msg.get("content", "") or ""
        if "error" in data:
            raise RuntimeError(f"Argo error: {data['error']}")
    raise RuntimeError(f"unexpected Argo response shape: {str(data)[:200]}")


def parse_embed_response(data: Any, n: int) -> list[list[float]]:
    if isinstance(data, dict):
        if isinstance(data.get("embedding"), list):
            vecs = data["embedding"]
        elif isinstance(data.get("data"), list):  # OpenAI shape
            vecs = [d["embedding"] for d in data["data"]]
        else:
            raise RuntimeError(f"unexpected Argo embed response: {str(data)[:200]}")
        if n == 1 and vecs and isinstance(vecs[0], (int, float)):
            vecs = [vecs]
        if len(vecs) != n:
            raise RuntimeError(f"Argo returned {len(vecs)} embeddings for {n} inputs")
        return vecs
    raise RuntimeError(f"unexpected Argo embed response: {str(data)[:200]}")


class ArgoLLM(LLM):
    provider = "argo"

    def __init__(
        self,
        model: str,
        *,
        user: str | None = None,
        user_env: str = "ARGO_USER",
        env: str = "prod",
        base_url: str | None = None,
        timeout: float = 120.0,
        transport: Any = None,  # httpx transport override (tests)
        **kwargs: Any,
    ):
        kwargs.pop("api_key_env", None)
        kwargs.pop("api_key", None)
        super().__init__(model, **kwargs)
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError("httpx is required for the argo provider (uv add httpx)") from exc
        self.user = resolve_user(user, user_env)
        self.url = resolve_base(env, base_url) + CHAT_PATH
        self._client = httpx.AsyncClient(transport=transport, **_client_kwargs(timeout))

    async def _complete(self, messages, *, temperature, json_mode, task, trial, max_tokens):
        payload = build_chat_payload(
            user=self.user,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode and not _REASONING.match(self.model):
            # No response_format on the native endpoint; rely on the schema instruction + repair loop.
            pass
        resp = await self._client.post(self.url, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"Argo HTTP {resp.status_code}: {resp.text[:300]}")
        return parse_chat_response(resp.json())


class ArgoEmbedder(Embedder):
    """Argo embeddings: models ``ada002``, ``v3small`` (default), ``v3large``."""

    name = "argo"

    def __init__(
        self,
        model: str = "v3small",
        *,
        user: str | None = None,
        user_env: str = "ARGO_USER",
        env: str = "prod",
        base_url: str | None = None,
        cache: ResponseCache | None = None,
        batch_size: int = 16,
        timeout: float = 120.0,
        transport: Any = None,
        max_concurrency: int = 4,
    ):
        import httpx

        self.model = model
        self.user = resolve_user(user, user_env)
        self.url = resolve_base(env, base_url) + EMBED_PATH
        self.cache = cache
        self.batch_size = batch_size
        self._client = httpx.AsyncClient(transport=transport, **_client_kwargs(timeout))
        self._sem = asyncio.Semaphore(max_concurrency)

    def _key(self, text: str) -> str:
        return stable_hash({"argo_emb": self.model, "text": text})

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self._sem:
            resp = await self._client.post(
                self.url, json={"user": self.user, "model": self.model, "prompt": texts}
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"Argo embed HTTP {resp.status_code}: {resp.text[:300]}")
        return parse_embed_response(resp.json(), len(texts))

    async def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float] | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            hit = self.cache.get(self._key(t)) if self.cache else None
            if hit is not None:
                out[i] = json.loads(hit)
            else:
                todo.append(i)
        batches = [todo[s : s + self.batch_size] for s in range(0, len(todo), self.batch_size)]
        results = await asyncio.gather(*(self._embed_batch([texts[i] for i in b]) for b in batches))
        for idx, vecs in zip(batches, results, strict=True):
            for i, v in zip(idx, vecs, strict=True):
                out[i] = v
                if self.cache:
                    self.cache.set(self._key(texts[i]), json.dumps(v))
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        arr = np.asarray(out, dtype=np.float32)
        self.dim = arr.shape[1]
        return _normalise(arr)
