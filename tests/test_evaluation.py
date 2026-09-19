import pytest
from conftest import mock_config

from trivium.config import Difficulty
from trivium.evaluation.difficulty import calibrate
from trivium.evaluation.evaluator import Evaluator
from trivium.generation.atomic import AtomicGenerator
from trivium.generation.base import Candidate


def test_calibrate_tiers():
    assert (
        calibrate(
            {"small": 1.0, "medium": 1.0, "large": 1.0},
            threshold=0.8,
            closed_book_accuracy=0,
            reasoning_depth=1,
        )[0]
        == Difficulty.easy
    )
    assert (
        calibrate(
            {"small": 0.5, "medium": 1.0, "large": 1.0},
            threshold=0.8,
            closed_book_accuracy=0,
            reasoning_depth=1,
        )[0]
        == Difficulty.medium
    )
    assert (
        calibrate(
            {"small": 0.0, "medium": 0.5, "large": 1.0},
            threshold=0.8,
            closed_book_accuracy=0,
            reasoning_depth=1,
        )[0]
        == Difficulty.hard
    )
    assert (
        calibrate(
            {"small": 0.0, "medium": 0.0, "large": 0.5},
            threshold=0.8,
            closed_book_accuracy=0,
            reasoning_depth=1,
        )[0]
        == Difficulty.extreme
    )
    d, method = calibrate(
        {"large": 1.0}, threshold=0.8, closed_book_accuracy=1.0, reasoning_depth=1
    )
    assert d == Difficulty.easy and method == "proxy"
    assert (
        calibrate({"large": 1.0}, threshold=0.8, closed_book_accuracy=0.0, reasoning_depth=2)[0]
        == Difficulty.hard
    )


async def test_evaluator_accepts_grounded_candidate_and_rejects_duplicate(
    ctx, models, knowledge_base
):
    _, chunks, store = knowledge_base
    ev = Evaluator(ctx.config, models, store, chunks)
    cands = await AtomicGenerator(ctx).generate(3)
    first = await ev.evaluate(cands[0])
    assert first.evaluation.accepted, first.evaluation.rejection_reasons
    e = first.evaluation
    assert (
        e.open_book_accuracy == 1.0
        and e.closed_book_accuracy == 0.0
        and e.document_dependence == 1.0
    )
    assert e.evidence_entailment >= 0.9 and e.overall_quality >= 0.75
    assert e.retrieval_difficulty in ("easy", "medium", "hard", "very_hard")
    assert first.evidence and first.evidence[0].chunk_id in cands[0].source_chunk_ids
    assert first.provenance["prompt_version"]
    await ev.register_accepted(cands[0])
    dup = await ev.evaluate(cands[0].model_copy(update={"id": "dup"}))
    assert not dup.evaluation.accepted and "duplicate" in dup.evaluation.rejection_reasons
    assert dup.evaluation.max_similarity_to_existing_question > 0.99


async def test_strict_rag_mode_rejects_closed_book_answerable(models, knowledge_base):
    _, chunks, store = knowledge_base
    cfg = mock_config(generation={"require_document_dependence": True})
    ev = Evaluator(cfg, models, store, chunks)
    fact = next(f for f in store if f.subject == "XTB" and f.predicate == "developed_by")
    cand = Candidate(
        id="c1",
        question="What is the developed by of XTB?",
        answer="Grimme Group",
        source_fact_ids=[fact.id],
        source_chunk_ids=[fact.chunk_id],
        source_document_ids=[fact.document_id],
        strategy="atomic",
    )
    res = await ev.evaluate(cand)
    assert res.evaluation.closed_book_accuracy == 1.0
    assert "closed_book_too_easy" in res.evaluation.rejection_reasons


async def test_answer_leak_and_length_rejected(models, knowledge_base, ctx):
    _, chunks, store = knowledge_base
    ev = Evaluator(ctx.config, models, store, chunks)
    f = next(iter(store))
    leaky = Candidate(
        id="l",
        question=f"Is the value of {f.subject} {f.object.display()}?",
        answer=f.object.display(),
        source_fact_ids=[f.id],
        source_chunk_ids=[f.chunk_id],
        strategy="atomic",
    )
    res = await ev.evaluate(leaky)
    assert "answer_leaked_in_question" in res.evaluation.rejection_reasons
    long = leaky.model_copy(update={"id": "long", "question": "What?", "answer": "a " * 20})
    res = await ev.evaluate(long)
    assert "answer_too_long" in res.evaluation.rejection_reasons


@pytest.mark.parametrize("skill,expected", [(1.0, Difficulty.easy), (0.0, Difficulty.hard)])
async def test_difficulty_tiers_drive_labels(knowledge_base, skill, expected):
    from trivium.models.registry import ModelRegistry

    _, chunks, store = knowledge_base
    cfg = mock_config(
        models={
            "default": {"provider": "mock", "model": "mock-large"},
            "roles": {
                "difficulty_small": {"model": "mock-small", "mock_skill": skill},
                "difficulty_medium": {"model": "mock-medium", "mock_skill": skill},
            },
        }
    )
    models = ModelRegistry(cfg)
    assert set(models.difficulty_tiers()) == {"small", "medium", "large"}
    ev = Evaluator(cfg, models, store, chunks)
    f = next(x for x in store if x.subject == "MACE" and x.predicate == "mean_absolute_error")
    cand = Candidate(
        id="t",
        question="What is the mean absolute error of MACE?",
        answer="0.12 eV",
        answer_type="numeric",
        source_fact_ids=[f.id],
        source_chunk_ids=[f.chunk_id],
        strategy="atomic",
    )
    res = await ev.evaluate(cand)
    assert res.evaluation.accepted
    assert res.evaluation.difficulty == expected and res.evaluation.difficulty_method == "tiers"
    assert res.evaluation.tier_accuracy["large"] == 1.0


async def test_yes_no_answers_rejected_for_non_boolean(models, knowledge_base, ctx):
    _, chunks, store = knowledge_base
    ev = Evaluator(ctx.config, models, store, chunks)
    f = next(iter(store))
    cand = Candidate(
        id="yn",
        question="Can symbolic factorizations be reused?",
        answer="Yes",
        source_fact_ids=[f.id],
        source_chunk_ids=[f.chunk_id],
        strategy="atomic",
    )
    res = await ev.evaluate(cand)
    assert "yes_no_answer" in res.evaluation.rejection_reasons
