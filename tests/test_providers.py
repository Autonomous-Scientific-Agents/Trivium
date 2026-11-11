"""Hosted providers: Ask Sage and the ALCF inference endpoints."""

from __future__ import annotations

import pytest

from trivium.config import Config
from trivium.models.embeddings import build_embedder, with_retries
from trivium.models.llm import (
    ASKSAGE_BASE_URL,
    AskSageLLM,
    LLMError,
    OpenAILLM,
    ProviderAccessError,
    asksage_endpoint,
    build_llm,
)
from trivium.models.registry import ModelRegistry


@pytest.fixture
def sage_env(monkeypatch):
    for v in ("ASK_SAGE_API", "ASK_SAGE_API_KEY", "ASKSAGE_API_KEY", "ASK_SAGE_BASE_URL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leak")
    return monkeypatch


def test_endpoint_defaults_to_anl_and_never_uses_the_openai_key(sage_env):
    with pytest.raises(ValueError, match="ASK_SAGE_API"):
        asksage_endpoint(None, "OPENAI_API_KEY")
    sage_env.setenv("ASK_SAGE_API", "sage-key")
    assert asksage_endpoint(None, "OPENAI_API_KEY") == (ASKSAGE_BASE_URL, "sage-key")
    sage_env.setenv("MY_SAGE", "other")
    assert asksage_endpoint("https://x/v1", "MY_SAGE") == ("https://x/v1", "other")
    sage_env.setenv("ASK_SAGE_BASE_URL", "https://api.asksage.ai/server/openai/v1")
    assert asksage_endpoint()[0] == "https://api.asksage.ai/server/openai/v1"


def test_registry_builds_asksage_models_and_embeddings(sage_env, tmp_path):
    sage_env.setenv("ASK_SAGE_API", "sage-key")
    cfg = Config.load(
        "examples/config.asksage.yaml", cache={"enabled": False}, output=str(tmp_path / "q.json")
    )
    reg = ModelRegistry(cfg)
    gen = reg["question_generator"]
    assert isinstance(gen, AskSageLLM) and gen.provider == "asksage"
    assert str(gen._client.base_url).rstrip("/") == ASKSAGE_BASE_URL
    assert gen._client.api_key == "sage-key"
    assert reg["difficulty_small"].model == "gpt-5.4-nano"
    assert reg.embedder.name == "asksage" and reg.embedder.model == "text-embedding-3-large"
    assert reg.embedder._client.api_key == "sage-key"


class _Resp:
    class _Choice:
        class message:  # noqa: N801 - mimics the openai object
            content = '{"ok": true}'

    choices = [_Choice()]


class _Completions:
    def __init__(self, error=None):
        self.error, self.kwargs, self.calls = error, None, 0

    async def create(self, **kw):
        self.calls += 1
        self.kwargs = kw
        if self.error:
            raise self.error
        return _Resp()


def _with_fake_client(llm: OpenAILLM, error=None) -> _Completions:
    comp = _Completions(error)
    llm._client = type("C", (), {"chat": type("Ch", (), {"completions": comp})()})()
    return comp


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-5.6-terra", "gpt-5.4-nano", "gpt-o3"])
async def test_reasoning_models_get_max_completion_tokens(sage_env, model):
    sage_env.setenv("ASK_SAGE_API", "k")
    llm = build_llm("asksage", model, max_tokens=123)
    comp = _with_fake_client(llm)
    await llm.generate([{"role": "user", "content": "hi"}])
    assert comp.kwargs["max_completion_tokens"] == 123
    assert "max_tokens" not in comp.kwargs and "temperature" not in comp.kwargs


async def test_non_reasoning_models_keep_sampling_parameters(sage_env):
    sage_env.setenv("ASK_SAGE_API", "k")
    llm = build_llm("asksage", "gpt-4.1", max_tokens=50)
    comp = _with_fake_client(llm)
    await llm.generate([{"role": "user", "content": "hi"}])
    assert comp.kwargs["max_tokens"] == 50 and comp.kwargs["temperature"] == 0.0


class RateLimitError(Exception):
    status_code = 429


async def test_quota_error_stops_without_retrying(sage_env):
    sage_env.setenv("ASK_SAGE_API", "k")
    llm = build_llm("asksage", "gpt-6-luna", cache=None)
    comp = _with_fake_client(llm, RateLimitError("Token quota exceeded for the user / plan"))
    with pytest.raises(ProviderAccessError, match="quota exceeded"):
        await llm.generate([{"role": "user", "content": "hi"}])
    assert comp.calls == 1


async def test_other_transient_errors_are_still_retried(sage_env):
    sage_env.setenv("ASK_SAGE_API", "k")
    llm = build_llm("asksage", "gpt-6-luna", cache=None)
    llm.max_retries = 2
    comp = _with_fake_client(llm, RateLimitError("Too many requests, slow down"))
    with pytest.raises(LLMError):
        await llm.generate([{"role": "user", "content": "hi"}])
    assert comp.calls == 2


async def test_embedding_quota_error_stops(sage_env):
    calls = []

    async def fail(**kw):
        calls.append(kw)
        raise RateLimitError("API key has been revoked or expired.")

    with pytest.raises(ProviderAccessError):
        await with_retries(fail, attempts=5)
    assert len(calls) == 1


def test_asksage_embedder_needs_a_key(sage_env):
    with pytest.raises(ValueError):
        build_embedder("asksage")


# ------------------------------------------------------------ ALCF inference endpoints
@pytest.fixture
def alcf_env(monkeypatch):
    import trivium.models.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_alcf_token", None)
    monkeypatch.setenv("ALCF_ACCESS_TOKEN", "globus-token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leak")
    return monkeypatch


def test_alcf_cluster_urls():
    from trivium.models.llm import alcf_base_url

    base = "https://inference-api.alcf.anl.gov/resource_server"
    assert alcf_base_url() == f"{base}/sophia/vllm/v1"
    assert alcf_base_url("Minerva") == f"{base}/minerva/api/v1"
    with pytest.raises(ValueError, match="unknown ALCF cluster"):
        alcf_base_url("polaris")


def test_registry_builds_alcf_models_per_cluster(alcf_env, tmp_path):
    from trivium.models.llm import ALCFLLM

    cfg = Config.load(
        "examples/config.alcf.yaml", cache={"enabled": False}, output=str(tmp_path / "q.json")
    )
    reg = ModelRegistry(cfg)
    gen, judge = reg["question_generator"], reg["judge"]
    assert isinstance(gen, ALCFLLM) and gen.cluster == "sophia"
    assert str(gen._client.base_url).rstrip("/").endswith("/sophia/vllm/v1")
    assert judge.cluster == "minerva" and judge.model == "mistral-large-4-nvfp4"
    assert gen._client.api_key == "globus-token"
    assert reg.embedder.name == "alcf" and reg.embedder.model == "google/embeddinggemma-300m"


async def test_alcf_options_and_parameters(alcf_env):
    llm = build_llm("alcf", "openai/gpt-oss-120b", reasoning_effort="low", max_tokens=99)
    comp = _with_fake_client(llm)
    await llm.generate([{"role": "user", "content": "hi"}])
    assert comp.kwargs["reasoning_effort"] == "low"
    assert comp.kwargs["max_tokens"] == 99 and "max_completion_tokens" not in comp.kwargs


class AuthenticationError(Exception):
    status_code = 401


async def test_alcf_refreshes_an_expired_token_once(alcf_env):
    import trivium.models.llm as llm_mod

    llm = build_llm("alcf", "openai/gpt-oss-20b", cache=None)
    seen = []

    class Flaky(_Completions):
        async def create(self, **kw):
            seen.append(llm._client.api_key)
            if len(seen) == 1:
                raise AuthenticationError("token expired")
            return _Resp()

    comp = Flaky()
    client = type("C", (), {"chat": type("Ch", (), {"completions": comp})(), "api_key": "old"})()
    llm._client = client
    alcf_env.delenv("ALCF_ACCESS_TOKEN")
    alcf_env.setattr(
        llm_mod, "alcf_token", lambda env=None, refresh=False: "fresh" if refresh else "x"
    )
    await llm.generate([{"role": "user", "content": "hi"}])
    assert seen == ["old", "fresh"]


def test_alcf_without_a_token_explains_how_to_log_in(monkeypatch):
    import shutil

    import trivium.models.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_alcf_token", None)
    monkeypatch.delenv("ALCF_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="alcf-tokens login inference"):
        llm_mod.alcf_token()


# ------------------------------------------------------------------- mixed providers
def test_role_switching_provider_drops_default_endpoint_and_options():
    from trivium.config import ModelsConfig

    models = ModelsConfig.model_validate(
        {
            "default": {
                "provider": "openai",
                "base_url": "http://127.0.0.1:44497/v1",
                "api_key_env": "ARGO_USER",
                "model": "gpt41",
                "max_tokens": 3000,
                "options": {"user": "me"},
            },
            "roles": {
                "judge": {"provider": "alcf", "model": "m", "options": {"cluster": "tara"}},
                "open_book": {"model": "gpt41mini"},  # same provider: inherits everything
            },
        }
    )
    judge = models.spec_for("judge")
    assert judge.base_url is None and judge.api_key_env == "OPENAI_API_KEY"
    assert judge.options == {"cluster": "tara"} and judge.max_tokens == 3000
    ob = models.spec_for("open_book")
    assert ob.base_url == "http://127.0.0.1:44497/v1" and ob.options == {"user": "me"}


def test_mixed_example_config(alcf_env, sage_env, tmp_path):
    from trivium.models.llm import ALCFLLM

    sage_env.setenv("ASK_SAGE_API", "sage-key")
    cfg = Config.load(
        "examples/config.mixed.yaml", cache={"enabled": False}, output=str(tmp_path / "q.json")
    )
    reg = ModelRegistry(cfg)
    assert isinstance(reg["question_generator"], ALCFLLM)
    assert isinstance(reg["judge"], AskSageLLM) and reg["judge"]._client.api_key == "sage-key"
    assert str(reg["judge"]._client.base_url).rstrip("/") == ASKSAGE_BASE_URL
    assert reg["open_book"].cluster == "minerva"
    assert reg.embedder.name == "alcf"
