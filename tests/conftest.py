from __future__ import annotations

import random
from pathlib import Path

import pytest

from trivium.config import Config
from trivium.generation.base import GenerationContext
from trivium.graph.builder import build_graph
from trivium.models.registry import ModelRegistry
from trivium.orchestration.controller import build_knowledge_base, load_documents

DATA = Path(__file__).parent / "data"


def mock_config(**overrides) -> Config:
    base = {
        "inputs": [str(DATA)],
        "models": {
            "default": {"provider": "mock", "model": "mock-large"},
            "embeddings": {"provider": "hashing"},
        },
        "cache": {"enabled": False},
        "generation": {"seed": 11, "batch_size": 6},
        "log_level": "WARNING",
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return Config.model_validate(base)


@pytest.fixture(scope="session")
def config() -> Config:
    return mock_config()


@pytest.fixture(scope="session")
def models(config) -> ModelRegistry:
    return ModelRegistry(config)


@pytest.fixture(scope="session")
def knowledge_base(config, models):
    """(documents, chunks, store) built once from tests/data with the mock backend."""
    import asyncio

    async def _build():
        docs = await load_documents(config)
        chunks, store = await build_knowledge_base(config, models, docs)
        return docs, chunks, store

    return asyncio.run(_build())


@pytest.fixture
def ctx(config, models, knowledge_base) -> GenerationContext:
    docs, chunks, store = knowledge_base
    return GenerationContext(
        config=config,
        store=store,
        graph=build_graph(store),
        chunks={c.id: c for c in chunks},
        llm=models["question_generator"],
        rng=random.Random(1),
    )
