"""Regression tests for the findings in review.md (2026-09-19)."""

from __future__ import annotations

import random

import numpy as np
import pytest
from conftest import mock_config

from trivium.config import ChunkingConfig, Config, ModelSpec
from trivium.evaluation.evaluator import Evaluator
from trivium.evaluation.grading import match_answer, normalize_answer
from trivium.generation.base import GeneratedQuestion, GenerationContext, answers_match
from trivium.generation.derived import _fmt
from trivium.generation.true_false import perturb_number
from trivium.graph.builder import build_graph
from trivium.ingestion.chunker import chunk_document
from trivium.ingestion.models import Document
from trivium.ingestion.parser import parse_text
from trivium.models.embeddings import Embedder, with_retries
from trivium.models.llm import LLM, LLMError, LLMResponse, is_retryable
from trivium.models.registry import ModelRegistry
from trivium.orchestration.controller import generate_dataset


# ----------------------------------------------------------------------------- 1.1
def test_choice_letter_a_matches():
    assert normalize_answer("A") == "a"
    assert match_answer("A", ["A", "Paris"], "choice")
    assert match_answer("A) Paris", ["A", "Paris"], "choice")
    assert match_answer("a", ["A", "Paris"], "choice")
    assert not match_answer("C", ["A", "Paris"], "choice")


async def test_pipeline_produces_multiple_choice_with_answer_a(tmp_path):
    cfg = mock_config(
        output=str(tmp_path / "q.json"),
        generation={"num_questions": 12, "question_types": ["multiple_choice"], "seed": 5},
    )
    ds, _ = await generate_dataset(cfg, write=False)
    mc = [q for q in ds.questions if q.type == "multiple_choice"]
    assert len(mc) >= 8
    letters = {q.answer for q in mc}
    assert "A" in letters, letters
    for q in mc:
        assert q.evaluation.open_book_accuracy == 1.0


# ----------------------------------------------------------------------------- 1.2
def test_fmt_keeps_integer_zeros():
    assert _fmt(1000) == "1000"
    assert _fmt(1200) == "1200"
    assert _fmt(250000) == "250000"
    assert _fmt(0.97) == "0.97"
    assert _fmt(3.0) == "3"
    assert _fmt(0.0) == "0"


# ----------------------------------------------------------------------------- 1.3
def test_answers_match_is_token_based():
    assert not answers_match("2.5", "12.5")
    assert not answers_match("1", "10")
    assert not answers_match("no", "know")
    assert answers_match("0.12", "0.12 eV")
    assert answers_match("Institution Y", "institution y")


class _FixedLLM(LLM):
    """Returns a canned GeneratedQuestion regardless of the prompt."""

    provider = "fixed"

    def __init__(self, answer: str):
        super().__init__("fixed", cache=None)
        self._answer = answer

    async def generate(self, messages, response_model=None, **kwargs):
        parsed = GeneratedQuestion(question="What is the derived value?", answer=self._answer)
        return LLMResponse(
            text=parsed.model_dump_json(), parsed=parsed, cached=False, model="fixed"
        )


async def test_ask_llm_never_overrides_fixed_answer(knowledge_base, config):
    from trivium.generation.atomic import AtomicGenerator

    _, chunks, store = knowledge_base
    ctx = GenerationContext(
        config=config,
        store=store,
        graph=build_graph(store),
        chunks={c.id: c for c in chunks},
        llm=_FixedLLM("2.5"),
        rng=random.Random(0),
    )
    gen = AtomicGenerator(ctx)
    fact = next(iter(store))
    gq = await gen.ask_llm([fact], "irrelevant", fixed_answer="12.5")
    assert gq is not None and gq.answer == "12.5"
    assert "2.5" not in gq.accepted_answers


# ----------------------------------------------------------------------------- 2.1
@pytest.mark.parametrize(
    "prediction,reference,answer_type",
    [
        ("Table 4", "Figure 4", "short_answer"),
        ("12 miles", "12 km", "numeric"),
        ("Paris or London", "Paris", "short_answer"),
        ("not Paris", "Paris", "short_answer"),
        ("2023", "2021", "numeric"),
        ("XTB had the highest runtime among MACE and XTB", "MACE", "short_answer"),
    ],
)
def test_grading_rejects_hedges_units_and_lookalikes(prediction, reference, answer_type):
    assert not match_answer(prediction, [reference], answer_type)


@pytest.mark.parametrize(
    "prediction,reference,answer_type",
    [
        ("QM9X covers molecules with up to 9 heavy atoms", "9 heavy atoms", "numeric"),
        ("DFT with the wB97X functional", "DFT", "short_answer"),
        ("14 h", "14 hours", "numeric"),
        ("50%", "50 percent", "numeric"),
        ("134,000 molecules", "134000", "numeric"),
    ],
)
def test_grading_accepts_equivalents(prediction, reference, answer_type):
    assert match_answer(prediction, [reference], answer_type)


# ----------------------------------------------------------------------------- 2.2 / 2.3
def test_is_retryable_classification():
    assert not is_retryable(ValueError("bad"))
    assert not is_retryable(RuntimeError("Argo HTTP 401: unauthorized"))
    assert not is_retryable(RuntimeError("Error code: 400 - invalid_request_error"))
    assert is_retryable(RuntimeError("Argo HTTP 503: service unavailable"))
    assert is_retryable(TimeoutError("timed out"))


async def test_with_retries_recovers_from_transient_failure():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("HTTP 502 bad gateway")
        return "ok"

    assert await with_retries(flaky, attempts=3) == "ok"
    assert calls["n"] == 3

    async def broken():
        raise RuntimeError("HTTP 401 unauthorized")

    with pytest.raises(LLMError):
        await with_retries(broken, attempts=3)


class _BrokenEmbedder(Embedder):
    name = "broken"

    async def embed(self, texts):
        raise LLMError("embedding backend down")


class _HashThenBroken(Embedder):
    """Works for the first call (registering) and fails afterwards."""

    name = "flaky"

    def __init__(self):
        from trivium.models.embeddings import HashingEmbedder

        self._inner = HashingEmbedder()
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        if self.calls > 1:
            raise LLMError("embedding backend down")
        return await self._inner.embed(texts)


async def test_embedder_failure_yields_evaluation_error_not_crash(ctx, models, knowledge_base):
    from trivium.generation.atomic import AtomicGenerator

    _, chunks, store = knowledge_base
    ev = Evaluator(ctx.config, models, store, chunks)
    cands = await AtomicGenerator(ctx).generate(2)
    await ev.register_accepted(cands[0])  # index now non-empty → assess() embeds
    ev.redundancy.embedder = _BrokenEmbedder()
    res = await ev.evaluate(cands[1])
    assert not res.evaluation.accepted
    assert res.evaluation.rejection_reasons == ["evaluation_error"]
    assert res.evaluation.error and "down" in res.evaluation.error
    assert res.evaluation.closed_book_accuracy is None  # nothing was misreported as a wrong answer


# ----------------------------------------------------------------------------- 2.5 / 2.7 / 2.8
def test_env_does_not_override_cli(monkeypatch, tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("models:\n  default: {provider: mock, model: from_yaml}\n")
    monkeypatch.setenv("TRIVIUM_MODEL", "from_env")
    assert Config.load(y).models.default.model == "from_env"  # env beats yaml
    cfg = Config.load(y, models={"default": {"model": "from_cli"}})
    assert cfg.models.default.model == "from_cli"  # cli beats env


def test_model_spec_key_and_merge():
    a = ModelSpec(provider="argo", model="gpt41", max_tokens=1024)
    b = ModelSpec(provider="argo", model="gpt41", max_tokens=4096)
    assert a.key() != b.key()
    base = ModelSpec(provider="argo", model="gpt41", options={"user": "u", "env": "prod"})
    merged = base.merged({"model": "gpt41mini", "options": {"env": "dev"}})
    assert merged.model == "gpt41mini" and merged.options == {"user": "u", "env": "dev"}
    with pytest.raises(ValueError):
        base.merged({"temperature": "hot"})


def test_public_dump_redacts_user():
    cfg = mock_config(
        models={"default": {"provider": "mock", "options": {"user": "keceli", "env": "prod"}}}
    )
    dumped = cfg.public_dump()
    assert dumped["models"]["default"]["options"] == {"user": "<redacted>", "env": "prod"}
    assert "inputs" not in dumped


def test_quota_sum_must_fit_num_questions():
    with pytest.raises(ValueError):
        mock_config(generation={"num_questions": 5, "easy": 4, "hard": 4})


# ----------------------------------------------------------------------------- 2.9 / parser
def test_chunk_offsets_are_exact_for_split_paragraphs():
    sents = [f"Sentence number {i} talks about\nsome topic {i}." for i in range(40)]
    raw = "# Title\n\n" + " ".join(sents) + "\n\n## Next\n\nShort paragraph."
    text, sections, _ = parse_text(raw, doc_id="d", fallback_title="t")
    doc = Document(id="d", title="t", source="mem", text=text, sections=sections)
    chunks = chunk_document(doc, ChunkingConfig(target_chars=300, max_chars=400, min_chars=20))
    assert len(chunks) > 3
    for c in chunks:
        # every chunk's first paragraph must be locatable verbatim at its recorded offset
        first = c.text.split("\n\n")[0]
        assert doc.text[c.start_char : c.start_char + len(first)] == first, c.id
        assert len(c.text) <= 400 + 60


def test_single_long_sentence_is_split():
    raw = "word " * 300
    text, sections, _ = parse_text(raw, doc_id="d", fallback_title="t")
    doc = Document(id="d", title="t", source="mem", text=text, sections=sections)
    chunks = chunk_document(doc, ChunkingConfig(target_chars=200, max_chars=250, min_chars=20))
    assert len(chunks) >= 5 and all(len(c.text) <= 250 for c in chunks)


def test_headings_inside_code_fences_are_ignored_and_ids_unique():
    raw = "# Preamble\n\nintro\n\n```python\n# not a heading\nx = 1\n```\n\n## Methods\n\nbody"
    text, sections, _ = parse_text("intro text\n\n" + raw, doc_id="d", fallback_title="t")
    ids = [s.id for s in sections]
    assert "not_a_heading" not in "".join(ids)
    assert len(ids) == len(set(ids))
    assert ids[0] == "preamble" and "preamble_2" in ids


# ----------------------------------------------------------------------------- misc
def test_perturb_keeps_percentages_and_integers_sane():
    rng = random.Random(1)
    for _ in range(20):
        assert 0 <= perturb_number(90.0, rng, text="90", unit="%") <= 100
        assert float(perturb_number(3, rng, text="3", unit="runs")).is_integer()


def test_multihop_mentions_uses_token_boundaries():
    from trivium.generation.multihop import _mentions

    assert not _mentions("What happened in 2015?", "5")
    assert not _mentions("Which dataset trained ANI-1x?", "ANI")
    assert _mentions("Who developed MACE in 2015?", "MACE")


async def test_registry_aclose_is_safe(config):
    reg = ModelRegistry(config)
    await reg.aclose()
    reg.close()
    assert isinstance(
        np.zeros(1), np.ndarray
    )  # keep numpy import meaningful for the embedder tests
