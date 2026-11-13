"""Personas, gap-driven questions, and the correctness fixes from the first Aurora runs."""

from __future__ import annotations

import random

import pytest
from conftest import mock_config

from trivium.cli import _overrides, build_parser
from trivium.evaluation.form import answer_grounded, content_tokens, lexical_overlap
from trivium.facts.models import Fact, FactObject, is_setting
from trivium.facts.store import FactStore
from trivium.generation.base import PERSONA_QUOTE, Candidate, GenerationContext
from trivium.generation.comparison import comparable_groups
from trivium.generation.multiple_choice import MultipleChoiceConverter, _list_items
from trivium.generation.persona import Ideas, PersonaGenerator, _in_text, outline
from trivium.graph.builder import build_graph
from trivium.models.llm import MockLLM, ProviderAccessError, access_denied
from trivium.models.registry import ModelRegistry
from trivium.orchestration.controller import generate_dataset
from trivium.output.feedback import auc, report

NEW_USER = "a new ALCF user who just received their first allocation"


def _fact(i, subject, predicate, obj, value=None, chunk="c1"):
    return Fact(
        id=f"f{i}",
        subject=subject,
        predicate=predicate,
        object=FactObject(text=obj, value=value),
        document_id="d",
        chunk_id=chunk,
        evidence=f"{subject} {predicate} {obj}",
    )


# ------------------------------------------------------- configuration values
def test_settings_are_detected():
    assert is_setting(_fact(1, "NRANKS_PER_NODE", "set_to", "12", 12))
    assert is_setting(_fact(2, "NTHREADS", "set_to", "1", 1))
    assert is_setting(_fact(3, "--ppn", "takes", "12", 12))
    assert not is_setting(_fact(4, "Aurora compute node", "gpus_per_node", "6", 6))
    assert not is_setting(_fact(5, "MACE", "has_error", "0.23 eV", 0.23))
    assert not is_setting(_fact(6, "Boltzmann constant", "value", "1.38e-23 J/K", 1.38e-23))
    assert not is_setting(_fact(7, "MACE MAE", "has_value", "0.12 eV", 0.12))


def test_settings_are_not_compared():
    facts = [
        _fact(1, "NRANKS_PER_NODE", "set_to", "12", 12),
        _fact(2, "NTHREADS", "set_to", "1", 1),
        _fact(3, "Polaris", "gpus_per_node", "4", 4),
        _fact(4, "Aurora", "gpus_per_node", "6", 6),
    ]
    groups = comparable_groups(facts)
    assert [[f.subject for f in g] for g in groups] == [["Polaris", "Aurora"]]


# --------------------------------------------- multiple choice: listed items
def test_items_listed_with_the_answer_are_not_distractors(config, models):
    store = FactStore(
        [
            _fact(1, "pip", "installs_package", "numpy"),
            _fact(2, "pip", "installs_packages", "pip, wheel, setuptools and cython"),
            _fact(3, "conda", "installs_package", "scipy"),  # other subject: fair distractor
        ]
    )
    ctx = GenerationContext(
        config, store, build_graph(store), {}, models["question_generator"], random.Random(0)
    )
    cand = Candidate(
        id="c",
        question="Which package does pip install?",
        answer="numpy",
        strategy="atomic",
        source_fact_ids=["f1"],
    )
    listed = MultipleChoiceConverter(ctx)._listed_with_answer(cand)
    assert {"wheel", "setuptools", "cython", "numpy"} <= set(listed)
    assert "scipy" not in listed
    assert _list_items("a, b and c; d") == ["a", "b", "c", "d"]


# --------------------------------------------------------- form helpers
def test_answer_grounding_and_overlap():
    ev = ["Support for building LAMMPS via Makefiles will be deprecated soon in favor of CMake."]
    assert answer_grounded("Makefiles", ev)
    assert answer_grounded("Makefile", ev)  # plural tolerance
    assert not answer_grounded("login nodes", ev)
    assert answer_grounded("16000 rows", ["These matrices have 16,000 and 64,000 rows"])
    assert content_tokens("What is the 16,000-row matrix?") == ["16000-row", "matrix"]
    assert lexical_overlap("Which build method for LAMMPS is being deprecated?", ev) == 0.4
    assert lexical_overlap("Which build method for LAMMPS is being deprecated?", []) == 0.0


def test_quote_matching_ignores_markdown_and_spacing():
    text = "Use `module load   apptainer` on a **compute** node."
    assert _in_text("Use module load apptainer on a compute node", text)
    assert not _in_text("Use module load docker", text)


# ------------------------------------------------------------ persona strategy
def _persona_ctx(knowledge_base, **gen):
    docs, chunks, store = knowledge_base
    cfg = mock_config(mode="trivia", generation={"personas": [NEW_USER], **gen})
    models = ModelRegistry(cfg)
    return GenerationContext(
        config=cfg,
        store=FactStore(store.all()),
        graph=build_graph(store),
        chunks={c.id: c for c in chunks},
        llm=models["question_generator"],
        rng=random.Random(2),
        embedder=models.embedder,
    )


def test_outline_has_titles_and_sections(knowledge_base):
    ctx = _persona_ctx(knowledge_base)
    out = outline(ctx)
    assert out and all(d["sections"] for d in out)


async def test_persona_generator_grounds_questions_and_records_gaps(knowledge_base):
    ctx = _persona_ctx(knowledge_base)
    gen = PersonaGenerator(ctx)
    assert gen.available()
    cands = await gen.generate(8)
    assert cands
    for c in cands:
        assert c.strategy == "persona"
        assert c.derivation["persona"] == NEW_USER and c.derivation["style"]
        (f,) = ctx.store.many(c.source_fact_ids)
        assert f.derivation == PERSONA_QUOTE
        assert _in_text(f.evidence, ctx.chunks[f.chunk_id].text)  # verbatim quote
        assert answer_grounded(c.answer, [f.evidence])
    # quote facts never feed the fact-driven strategies
    assert all(f.derivation != PERSONA_QUOTE for f in ctx.sample_facts(200))
    assert any("allocation" in g["question"] for g in gen.gaps)
    assert all(g["status"] in ("unanswerable", "partial", "ungrounded") for g in gen.gaps)


def test_persona_generator_needs_personas(knowledge_base):
    docs, chunks, store = knowledge_base
    cfg = mock_config(mode="trivia")
    models = ModelRegistry(cfg)
    ctx = GenerationContext(
        cfg,
        store,
        build_graph(store),
        {c.id: c for c in chunks},
        models["question_generator"],
        random.Random(0),
        embedder=models.embedder,
    )
    assert not PersonaGenerator(ctx).available()


async def test_end_to_end_with_persona(tmp_path):
    cfg = mock_config(
        mode="trivia",
        output=str(tmp_path / "q.json"),
        generation={
            "num_questions": 10,
            "select_top": 4,
            "question_types": ["short_answer", "true_false"],
            "personas": [NEW_USER],
            "seed": 5,
        },
    )
    ds, run = await generate_dataset(cfg, write=False)
    assert ds.questions
    stats = ds.strategy_stats
    assert stats["persona"]["generated"] > 0 and stats["persona"]["accepted"] > 0
    assert stats["persona_gaps"]
    assert ds.dataset_metrics["average_lexical_overlap"] is not None
    assert all(q.evaluation.lexical_overlap is not None for q in ds.questions)


def test_cli_persona_flag():
    args = build_parser().parse_args(
        ["generate", "-i", "x.md", "--persona", NEW_USER, "--persona", "a PI"]
    )
    assert _overrides(args)["generation"]["personas"] == [NEW_USER, "a PI"]


# -------------------------------------------------------------- feedback report
def test_feedback_report_agreement():
    def row(rating, q, strategy="atomic"):
        return {
            "run": "r",
            "id": "q",
            "question": "?",
            "answer": "a",
            "type": "short_answer",
            "strategy": strategy,
            "style": None,
            "difficulty": "easy",
            "rating": rating,
            "comment": "too easy" if rating == "down" else "",
            "scores": {"overall_quality": q},
        }

    rows = [row("up", 0.9), row("up", 0.8, "persona"), row("down", 0.3), row("down", 0.85)]
    rep = report(rows)
    assert rep["overall"] == {"rated": 4, "up": 2, "down": 2, "up_rate": 0.5}
    assert rep["by_strategy"]["persona"]["up_rate"] == 1.0
    assert rep["score_agreement"]["overall_quality"]["auc"] == 0.75
    assert len(rep["down_comments"]) == 2
    assert auc([1.0], []) is None


# ----------------------------------------------------------- blocked provider
ARGO_NOTICE = (
    "⚠️ **IMPORTANT USAGE NOTICE FROM ARGO** ⚠️\n🚫 **ACCESS REVOKED** 🚫\n"
    "Your Argo usage limit has been exceeded. Reason: Monthly limit exceeded."
)


class _Blocked(MockLLM):
    async def _complete(self, messages, **kw):
        return ARGO_NOTICE


async def test_access_notice_stops_without_retrying():
    llm = _Blocked()
    with pytest.raises(ProviderAccessError, match="Monthly limit exceeded"):
        await llm.generate([{"role": "user", "content": "x"}], Ideas, task="persona_questions")
    assert llm.calls == 1  # no retries, no repair prompts
    assert access_denied("The answer is 42.") is None


async def test_access_notice_escapes_generators(knowledge_base):
    ctx = _persona_ctx(knowledge_base)
    ctx.llm = _Blocked()
    with pytest.raises(ProviderAccessError):
        await PersonaGenerator(ctx).generate(4)


async def test_access_notice_stops_the_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(MockLLM, "_complete", _Blocked._complete)
    cfg = mock_config(output=str(tmp_path / "q.json"), generation={"num_questions": 3})
    with pytest.raises(ProviderAccessError):
        await generate_dataset(cfg, write=False)
