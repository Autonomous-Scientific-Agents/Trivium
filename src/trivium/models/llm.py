"""Provider-independent LLM abstraction.

Backends:
    * ``openai``    — any OpenAI-compatible chat endpoint (OpenAI, vLLM, Ollama, ALCF, ...)
    * ``anthropic`` — Anthropic Messages API
    * ``mock``      — deterministic rule-based backend used for tests and dry runs

All backends implement :meth:`LLM.generate`, which returns either free text or an
instance of a Pydantic ``response_model`` parsed from the model's JSON output.
Responses are cached by content hash (see :mod:`trivium.models.cache`).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from trivium.models.cache import ResponseCache, stable_hash

T = TypeVar("T", bound=BaseModel)

Message = dict[str, str]


@dataclass
class LLMResponse:
    text: str
    parsed: BaseModel | None = None
    model: str = ""
    cached: bool = False
    usage: dict[str, int] = field(default_factory=dict)


_NON_RETRYABLE_STATUS = re.compile(r"\b(400|401|403|404|405|413|422)\b")
_NON_RETRYABLE_HINTS = (
    "invalid_api_key",
    "authentication",
    "unauthorized",
    "permission",
    "model_not_found",
    "does not exist",
    "unsupported_parameter",
    "invalid_request_error",
    "argo requires",
    "unknown argo env",
)


def is_retryable(exc: Exception) -> bool:
    """False for configuration/authorisation/validation errors that will not fix themselves."""
    if isinstance(exc, (ValueError, TypeError, KeyError, ImportError)):
        return False
    if exc.__class__.__name__ in (
        "ArgoConfigError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "BadRequestError",
        "UnprocessableEntityError",
    ):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status not in (400, 401, 403, 404, 405, 413, 422)
    msg = str(exc).lower()
    if any(h in msg for h in _NON_RETRYABLE_HINTS):
        return False
    return not _NON_RETRYABLE_STATUS.search(msg[:120])


class LLMError(RuntimeError):
    pass


class ProviderAccessError(RuntimeError):
    """The provider refused service (access revoked, usage limit exceeded).

    Deliberately *not* an :class:`LLMError`: every call site tolerates LLMErrors and moves on, but
    retrying or continuing against a blocked account only burns more requests. This propagates
    out of the pipeline and stops the run."""


# gateways that answer with a notice instead of an HTTP error (Argo: "ACCESS REVOKED ... Your
# Argo usage limit has been exceeded")
# Ask Sage: HTTP 429 "Token quota exceeded for the user / plan", "API key has been revoked or
# expired"; OpenAI: 429 "You exceeded your current quota")
_ACCESS_DENIED = re.compile(
    r"ACCESS REVOKED|usage limit (?:has been )?exceeded|quota (?:has been )?exceeded"
    r"|exceeded your current quota|revoked or expired",
    re.I,
)


def access_denied(text: str) -> str | None:
    """The provider's refusal notice, if ``text`` is one."""
    if _ACCESS_DENIED.search(text or ""):
        return re.sub(r"[*⚠️🚫]+", "", text).strip()[:500]
    return None


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _span(text: str) -> str | None:
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            return text[start : end + 1]
    return None


def extract_json(text: str) -> str:
    """Pull the JSON object/array out of a model response.

    Prefers a candidate that actually parses: the whole response, a fenced block, or the outermost
    brace span. A bare JSON response can contain fences of its own (e.g. a quoted ```bash block),
    so fences are not trusted blindly."""
    stripped = text.strip()
    fences = [m.group(1).strip() for m in _JSON_FENCE.finditer(text)]
    candidates = [stripped, *fences, _span(stripped), *(_span(f) for f in fences)]
    for c in candidates:
        if not c:
            continue
        try:
            json.loads(c)
        except ValueError:
            continue
        return c
    # nothing parses: best guess for the caller's error message / repair prompt
    return next((c for c in (_span(fences[0]) if fences else None, _span(stripped)) if c), stripped)


def _schema_instruction(response_model: type[BaseModel]) -> str:
    schema = json.dumps(response_model.model_json_schema(), indent=None)
    return (
        "Respond ONLY with a single JSON object that conforms to this JSON schema "
        "(no prose, no markdown fences):\n" + schema
    )


class LLM:
    """Base class. Subclasses implement :meth:`_complete`."""

    provider: str = "base"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache: ResponseCache | None = None,
        max_concurrency: int = 8,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(max_concurrency)
        self.calls = 0

    # ------------------------------------------------------------------ public
    async def aclose(self) -> None:
        """Release network resources (no-op for providers without a client)."""
        client = getattr(self, "_client", None)
        for meth in ("aclose", "close"):
            fn = getattr(client, meth, None)
            if fn is not None:
                res = fn()
                if asyncio.iscoroutine(res):
                    await res
                return

    async def generate(
        self,
        messages: list[Message],
        response_model: type[T] | None = None,
        *,
        temperature: float | None = None,
        task: str = "generic",
        trial: int = 0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run a chat completion.

        ``task`` is a free-form tag used for cache partitioning, logging and by the mock
        backend to select a behaviour; ``trial`` distinguishes repeated stochastic samples
        of the same prompt so that they are cached independently but reproducibly.
        """
        temp = self.temperature if temperature is None else temperature
        messages = list(messages)
        if response_model is not None:
            messages = messages + [
                {"role": "system", "content": _schema_instruction(response_model)}
            ]
        key = stable_hash(
            {
                "provider": self.provider,
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "response_model": response_model.__name__ if response_model else None,
                "task": task,
                "trial": trial,
                "max_tokens": max_tokens or self.max_tokens,
            }
        )
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return self._finish(hit, response_model, cached=True)

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with self._sem:  # hold the slot only for the call itself
                    text = await self._complete(
                        messages,
                        temperature=temp,
                        json_mode=response_model is not None,
                        task=task,
                        trial=trial,
                        max_tokens=max_tokens or self.max_tokens,
                    )
                self.calls += 1
            except Exception as exc:  # noqa: BLE001 — provider errors are heterogeneous
                last_err = exc
                notice = access_denied(str(exc))
                if notice:
                    raise ProviderAccessError(
                        f"{self.provider}:{self.model} refused service: {notice}"
                    ) from exc
                if not is_retryable(exc):
                    raise LLMError(
                        f"{self.provider}:{self.model} failed permanently: {exc}"
                    ) from exc
                await asyncio.sleep(min(2**attempt, 8))
                continue
            notice = access_denied(text)
            if notice:
                raise ProviderAccessError(f"{self.provider}:{self.model} refused service: {notice}")
            try:
                resp = self._finish(text, response_model, cached=False)
            except LLMError as exc:
                last_err = exc
                # Ask again with a repair hint appended.
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": "That was not valid JSON for the schema. Return only the JSON.",
                    },
                ]
                continue
            if self.cache is not None:
                self.cache.set(key, text)
            return resp
        raise LLMError(f"{self.provider}:{self.model} failed after retries: {last_err}")

    # ---------------------------------------------------------------- helpers
    def _finish(
        self, text: str, response_model: type[BaseModel] | None, *, cached: bool
    ) -> LLMResponse:
        parsed = None
        if response_model is not None:
            try:
                parsed = response_model.model_validate_json(extract_json(text))
            except (ValidationError, ValueError) as exc:
                raise LLMError(
                    f"could not parse response as {response_model.__name__}: {exc}"
                ) from exc
        return LLMResponse(text=text, parsed=parsed, model=self.model, cached=cached)

    async def _complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        json_mode: bool,
        task: str,
        trial: int,
        max_tokens: int,
    ) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"


# --------------------------------------------------------------------------- OpenAI
_OPENAI_REASONING = re.compile(r"^(o\d|gpt-o\d|gpt-[5-9])", re.I)


class OpenAILLM(LLM):
    """OpenAI-compatible chat completions (works with vLLM, Ollama, ALCF endpoints...)."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install 'trivium[openai]' to use the openai provider") from exc
        key = api_key or os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY") or "none"
        self._client = AsyncOpenAI(
            api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL")
        )
        self._json_mode_ok = True
        self._extra: dict[str, Any] = {}  # extra request fields, e.g. reasoning_effort

    async def _complete(self, messages, *, temperature, json_mode, task, trial, max_tokens):
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages, **self._extra)
        if _OPENAI_REASONING.match(self.model):
            # o-series / gpt-5 reject sampling parameters and use max_completion_tokens
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs.update(temperature=temperature, max_tokens=max_tokens)
        if trial:
            kwargs["seed"] = trial
        if json_mode and self._json_mode_ok:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if json_mode and self._json_mode_ok and "response_format" in str(exc):
                self._json_mode_ok = False  # server does not support JSON mode; fall back
                kwargs.pop("response_format", None)
                resp = await self._client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ""


# ------------------------------------------------------------------------- Ask Sage
ASKSAGE_BASE_URL = "https://api.asksage.anl.gov/server/openai/v1"  # ANL's instance
ASKSAGE_KEY_ENVS = ("ASK_SAGE_API", "ASK_SAGE_API_KEY", "ASKSAGE_API_KEY")


def asksage_endpoint(
    base_url: str | None = None, api_key_env: str | None = None
) -> tuple[str, str]:
    """Base URL and API key for Ask Sage's OpenAI-compatible API.

    The key comes from ``api_key_env`` or one of :data:`ASKSAGE_KEY_ENVS`, never from
    ``OPENAI_API_KEY`` (the generic default of ``api_key_env``), so an OpenAI key is not sent to
    Ask Sage. Other Ask Sage instances: set ``base_url`` or ``$ASK_SAGE_BASE_URL``, e.g.
    ``https://api.asksage.ai/server/openai/v1``.
    """
    envs = [api_key_env] if api_key_env and api_key_env != "OPENAI_API_KEY" else []
    envs += list(ASKSAGE_KEY_ENVS)
    key = next((os.environ[e] for e in envs if os.environ.get(e)), None)
    if not key:
        raise ValueError(f"Ask Sage needs an API key in ${envs[0]}")
    return base_url or os.environ.get("ASK_SAGE_BASE_URL") or ASKSAGE_BASE_URL, key


class AskSageLLM(OpenAILLM):
    """Ask Sage through its OpenAI-compatible endpoint (``Authorization: Bearer <api key>``)."""

    provider = "asksage"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        **kwargs: Any,
    ):
        url, key = asksage_endpoint(base_url, api_key_env)
        super().__init__(model, base_url=url, api_key=api_key or key, **kwargs)


# ------------------------------------------------------------- ALCF inference endpoints
ALCF_API = "https://inference-api.alcf.anl.gov/resource_server"
ALCF_CLUSTERS = {"sophia": "vllm", "metis": "api", "minerva": "api", "tara": "api"}
_alcf_token: str | None = None


def alcf_base_url(cluster: str = "sophia") -> str:
    """OpenAI-compatible base URL of an ALCF inference cluster."""
    cluster = cluster.lower()
    if cluster not in ALCF_CLUSTERS:
        raise ValueError(f"unknown ALCF cluster {cluster!r}; one of {sorted(ALCF_CLUSTERS)}")
    return f"{ALCF_API}/{cluster}/{ALCF_CLUSTERS[cluster]}/v1"


def alcf_token(api_key_env: str | None = None, *, refresh: bool = False) -> str:
    """A Globus access token for the ALCF inference service.

    Taken from ``api_key_env`` / ``$ALCF_ACCESS_TOKEN`` if set (a fixed token, never refreshed),
    otherwise from ``alcf-tokens get-token inference``, which refreshes expired tokens (they last
    48 hours) from the login cached by ``alcf-tokens login inference``.
    """
    global _alcf_token
    envs = [api_key_env] if api_key_env and api_key_env != "OPENAI_API_KEY" else []
    for env in [*envs, "ALCF_ACCESS_TOKEN"]:
        if os.environ.get(env):
            return os.environ[env]
    if _alcf_token and not refresh:
        return _alcf_token
    token = None
    try:
        from alcf_tokens.auth import get_access_token  # type: ignore[import-not-found]

        token = get_access_token("inference")
    except ImportError:
        import shutil
        import subprocess

        exe = shutil.which("alcf-tokens")
        if exe:
            out = subprocess.run(
                [exe, "get-token", "inference"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=120,
            )
            lines = out.stdout.strip().splitlines()
            token = lines[-1].strip() if out.returncode == 0 and lines else None
    if not token:
        raise ValueError(
            "no ALCF inference token: install alcf-tokens (uv tool install alcf-tokens), run "
            "`alcf-tokens login inference`, or set $ALCF_ACCESS_TOKEN"
        )
    _alcf_token = token
    return token


class ALCFLLM(OpenAILLM):
    """Open-weight models on the ALCF inference endpoints (Sophia, Metis, Minerva, Tara).

    ``options``: ``cluster`` (default ``sophia``) and ``reasoning_effort`` (for models with
    reasoning levels, e.g. ``low`` to save tokens)."""

    provider = "alcf"

    def __init__(
        self,
        model: str,
        *,
        cluster: str = "sophia",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ):
        self._api_key_env = api_key_env
        super().__init__(
            model,
            base_url=base_url or alcf_base_url(cluster),
            api_key=api_key or alcf_token(api_key_env),
            **kwargs,
        )
        self.cluster = cluster
        if reasoning_effort:
            self._extra["reasoning_effort"] = reasoning_effort

    async def _complete(self, messages, **kw):
        try:
            return await super()._complete(messages, **kw)
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "status_code", None) != 401:
                raise
            # the 48-hour token expired mid-run: fetch a refreshed one and try once more
            self._client.api_key = await asyncio.to_thread(
                alcf_token, self._api_key_env, refresh=True
            )
            return await super()._complete(messages, **kw)


# ------------------------------------------------------------------------ Anthropic
class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pip install 'trivium[anthropic]' to use the anthropic provider"
            ) from exc
        self._client = AsyncAnthropic(api_key=api_key or os.environ.get(api_key_env))

    async def _complete(self, messages, *, temperature, json_mode, task, trial, max_tokens):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        resp = await self._client.messages.create(
            model=self.model,
            system=system or None,
            messages=chat,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(getattr(block, "text", "") for block in resp.content)


# ----------------------------------------------------------------------------- Mock
class MockLLM(LLM):
    """Deterministic rule-based stand-in used for tests, dry runs and CI.

    It dispatches on the ``task`` tag and applies simple heuristics over the prompt
    (see :mod:`trivium.models.mock_rules`). It never calls the network.
    """

    provider = "mock"

    def __init__(self, model: str = "mock", *, skill: float = 1.0, **kwargs: Any):
        kwargs.setdefault("cache", None)
        super().__init__(model, **kwargs)
        self.skill = skill  # 0..1, probability-like ability used by answerer tasks

    async def _complete(self, messages, *, temperature, json_mode, task, trial, max_tokens):
        from trivium.models import mock_rules

        return mock_rules.respond(task, messages, skill=self.skill, trial=trial, model=self.model)


# -------------------------------------------------------------------------- factory
def build_llm(
    provider: str,
    model: str,
    *,
    cache: ResponseCache | None = None,
    **kwargs: Any,
) -> LLM:
    provider = provider.lower()
    if provider in ("openai", "openai-compatible", "vllm", "ollama"):
        return OpenAILLM(model, cache=cache, **kwargs)
    if provider in ("asksage", "ask-sage", "ask_sage"):
        return AskSageLLM(model, cache=cache, **kwargs)
    if provider == "alcf":
        return ALCFLLM(model, cache=cache, **kwargs)
    if provider == "anthropic":
        return AnthropicLLM(model, cache=cache, **kwargs)
    if provider == "argo":
        from trivium.models.argo import ArgoLLM

        return ArgoLLM(model, cache=cache, **kwargs)
    if provider == "mock":
        kwargs.pop("base_url", None)
        kwargs.pop("api_key_env", None)
        kwargs.pop("api_key", None)
        return MockLLM(model, cache=cache, **kwargs)
    raise ValueError(f"unknown LLM provider: {provider!r}")
