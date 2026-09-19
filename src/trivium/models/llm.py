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


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str:
    """Pull the first JSON object/array out of a model response."""
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return text


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
                if not is_retryable(exc):
                    raise LLMError(
                        f"{self.provider}:{self.model} failed permanently: {exc}"
                    ) from exc
                await asyncio.sleep(min(2**attempt, 8))
                continue
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
_OPENAI_REASONING = re.compile(r"^(o\d|gpt-5)", re.I)


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

    async def _complete(self, messages, *, temperature, json_mode, task, trial, max_tokens):
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages)
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
