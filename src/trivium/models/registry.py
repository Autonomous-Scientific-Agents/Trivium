"""Map configured model *roles* to LLM instances (design §29)."""

from __future__ import annotations

from trivium.config import Config, ModelSpec
from trivium.models.cache import ResponseCache
from trivium.models.embeddings import Embedder, build_embedder
from trivium.models.llm import LLM, build_llm

ROLES = (
    "fact_extractor",
    "fact_validator",
    "question_generator",
    "quality_evaluator",
    "judge",
    "closed_book",
    "open_book",
    "difficulty_small",
    "difficulty_medium",
    "difficulty_large",
)


class ModelRegistry:
    def __init__(self, config: Config):
        self.config = config
        self.cache = ResponseCache(config.cache_path) if config.cache.enabled else None
        self._llms: dict[str, LLM] = {}
        self._by_spec: dict[str, LLM] = {}
        for role in ROLES:
            spec = config.models.spec_for(role)
            if spec is None:
                continue
            self._llms[role] = self._get_or_build(spec)
        emb = config.models.embeddings
        emb_kwargs: dict = dict(emb.options)
        if emb.provider in ("openai", "argo") and emb.base_url:
            emb_kwargs["base_url"] = emb.base_url
        self.embedder: Embedder = build_embedder(
            emb.provider, emb.model, cache=self.cache, **emb_kwargs
        )

    def _get_or_build(self, spec: ModelSpec) -> LLM:
        key = spec.key()
        if key not in self._by_spec:
            self._by_spec[key] = build_llm(
                spec.provider,
                spec.model,
                cache=self.cache,
                temperature=spec.temperature,
                base_url=spec.base_url,
                api_key_env=spec.api_key_env,
                max_concurrency=spec.max_concurrency,
                max_tokens=spec.max_tokens,
                **({"skill": spec.mock_skill} if spec.provider == "mock" else {}),
                **spec.options,
            )
        return self._by_spec[key]

    def __getitem__(self, role: str) -> LLM:
        if role not in self._llms:
            raise KeyError(f"no model configured for role {role!r}")
        return self._llms[role]

    def get(self, role: str) -> LLM | None:
        return self._llms.get(role)

    def has(self, role: str) -> bool:
        return role in self._llms

    def difficulty_tiers(self) -> dict[str, LLM]:
        """Configured difficulty evaluators, ordered small → large."""
        tiers = {}
        for tier in ("small", "medium", "large"):
            llm = self._llms.get(f"difficulty_{tier}")
            if llm is not None:
                tiers[tier] = llm
        return tiers

    def describe(self) -> dict[str, str]:
        return {role: f"{llm.provider}:{llm.model}" for role, llm in self._llms.items()}

    def total_calls(self) -> int:
        return sum(llm.calls for llm in self._by_spec.values())

    async def aclose(self) -> None:
        """Close network clients (call from within the event loop, before :meth:`close`)."""
        for llm in self._by_spec.values():
            await llm.aclose()
        await self.embedder.aclose()

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()
