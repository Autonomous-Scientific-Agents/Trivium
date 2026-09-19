"""Argo provider tests using an in-process httpx transport (no network)."""

import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from trivium.config import Config
from trivium.models.argo import (
    ArgoConfigError,
    ArgoEmbedder,
    ArgoLLM,
    build_chat_payload,
    parse_chat_response,
    parse_embed_response,
)
from trivium.models.llm import build_llm
from trivium.models.registry import ModelRegistry
from trivium.prompts import build_messages

ARGO_CFG = Path(__file__).resolve().parent.parent / "examples" / "config.argo.yaml"


class Out(BaseModel):
    answer: str


def _transport(recorder: list):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        recorder.append((str(request.url), body))
        if request.url.path.endswith("/chat/"):
            return httpx.Response(
                200, json={"response": json.dumps({"answer": f"echo:{body['model']}"})}
            )
        if request.url.path.endswith("/embed/"):
            return httpx.Response(
                200, json={"embedding": [[1.0, 0.0, float(i)] for i in range(len(body["prompt"]))]}
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_payload_shapes():
    msgs = [{"role": "user", "content": "hi"}]
    p = build_chat_payload(user="u", model="gpt4o", messages=msgs, temperature=0.2, max_tokens=100)
    assert (
        p["user"] == "u"
        and p["temperature"] == 0.2
        and p["max_tokens"] == 100
        and "max_completion_tokens" not in p
    )
    for reasoning in ("gpto3", "gpto4mini", "gpt5", "gpt5mini", "o3"):
        r = build_chat_payload(
            user="u", model=reasoning, messages=msgs, temperature=0.2, max_tokens=100
        )
        assert "temperature" not in r and "top_p" not in r and r["max_completion_tokens"] == 100
    assert parse_chat_response({"response": "x"}) == "x"
    assert parse_chat_response({"choices": [{"message": {"content": "y"}}]}) == "y"
    with pytest.raises(RuntimeError):
        parse_chat_response({"error": "bad model"})
    assert parse_embed_response({"embedding": [0.1, 0.2]}, 1) == [[0.1, 0.2]]
    assert parse_embed_response({"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]}, 2) == [
        [1.0],
        [2.0],
    ]


def test_user_required(monkeypatch):
    monkeypatch.delenv("ARGO_USER", raising=False)
    with pytest.raises(ArgoConfigError):
        ArgoLLM("gpt4o")
    llm = ArgoLLM("gpt4o", user="someone", env="dev")
    assert llm.url == "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"


async def test_chat_roundtrip_and_cache(monkeypatch):
    monkeypatch.setenv("ARGO_USER", "tester")
    rec: list = []
    llm = build_llm("argo", "gpt41", transport=_transport(rec))
    msgs = build_messages("s", "i", {"q": 1})
    r = await llm.generate(msgs, Out, task="closed_book")
    assert r.parsed.answer == "echo:gpt41"
    url, body = rec[0]
    assert url.startswith("https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/")
    assert body["user"] == "tester" and body["model"] == "gpt41"
    assert (
        body["messages"][0]["role"] == "system" and body["messages"][-1]["role"] == "system"
    )  # schema instruction appended
    assert len(rec) == 1


async def test_embedder(monkeypatch):
    monkeypatch.setenv("ARGO_USER", "tester")
    rec: list = []
    emb = ArgoEmbedder(transport=_transport(rec), batch_size=2)
    vecs = await emb.embed(["a", "b", "c"])
    assert vecs.shape == (3, 3) and emb.dim == 3
    assert len(rec) == 2 and rec[0][1]["model"] == "v3small" and rec[0][1]["user"] == "tester"
    assert abs(float((vecs[0] ** 2).sum()) - 1) < 1e-5  # normalised


def test_argo_config_builds_registry(monkeypatch):
    monkeypatch.setenv("ARGO_USER", "tester")
    cfg = Config.load(ARGO_CFG, cache={"enabled": False})
    reg = ModelRegistry(cfg)
    desc = reg.describe()
    assert desc["fact_extractor"] == "argo:gpt41" and desc["difficulty_small"] == "argo:gpt41nano"
    assert set(reg.difficulty_tiers()) == {"small", "medium", "large"}
    assert reg.embedder.name == "argo" and reg.embedder.model == "v3small"
    # per-role env option changes the endpoint
    cfg2 = Config.load(
        ARGO_CFG,
        cache={"enabled": False},
        models={"roles": {"judge": {"model": "claudehaiku45", "options": {"env": "dev"}}}},
    )
    reg2 = ModelRegistry(cfg2)
    assert reg2["judge"].url.startswith("https://apps-dev.inside.anl.gov")
    assert reg2["judge"] is not reg2["open_book"]
