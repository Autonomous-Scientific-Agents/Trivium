"""Trivia-oriented generation: fact scoring, answer form, distractors, true/false edits, ranking."""

import random

import pytest
from conftest import mock_config

from trivium.config import Config, Difficulty
from trivium.evaluation.evaluator import EvaluatedCandidate, Evaluation, Evaluator, option_picks
from trivium.evaluation.form import clause_answer, generic_answer, local_label, source_reference
from trivium.facts.store import FactStore, norm
from trivium.facts.trivia import score_trivia
from trivium.generation.base import Candidate, GenerationContext
from trivium.generation.multiple_choice import (
    DistractorProposal,
    MultipleChoiceConverter,
    _pick,
    _stands_out,
)
from trivium.generation.true_false import TrueFalseGenerator
from trivium.graph.builder import build_graph
from trivium.models.registry import ModelRegistry
from trivium.orchestration import selection
from trivium.orchestration.controller import generate_dataset


# ------------------------------------------------------------------ form checks
@pytest.mark.parametrize(
    "question",
    [
        "Which matrix has the highest row count in the Methods section of doc_2018_keceli_siesta_sips?",
        "What value is reported in Table 3?",
        "What did the authors find?",
        "Which solver does the proposed method replace?",
        "Which procedure does the Aurora container guide describe?",
        "What does this documentation recommend for large jobs?",
        "Which topic is covered on this page?",
    ],
)
def test_source_references_are_detected(question):
    assert source_reference(question)


@pytest.mark.parametrize(
    "question",
    [
        "What was the simulation temperature in Run 2?",
        "Which has more rows, the larger matrix or the smaller matrix?",
        "What catalyst was used in Experiment 4?",
    ],
)
def test_local_labels_are_detected(question):
    assert local_label(question)


@pytest.mark.parametrize(
    "question",
    [
        "Which Group 14 element is used as the dopant?",
        "In which case a full diagonalisation is avoided?",
        "What figure of merit did MACE improve?",
        "Which institution created the QM9X dataset?",
        "Which restaurant rating is published by the Michelin Guide?",
        "What is the default page size on Aurora?",
    ],
)
def test_standalone_questions_pass_form_checks(question):
    assert not source_reference(question) and not local_label(question)


@pytest.mark.parametrize("answer", ["larger matrix", "either method", "important parameter"])
def test_generic_answers(answer):
    assert generic_answer(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "cases where eigenvalues coincide with slice boundaries",
        "only at the beginning of the SCF cycles",
        "includes gaps",
        "dataset of organic molecules",
    ],
)
def test_clause_answers(answer):
    assert clause_answer(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Thomas More",
        "Hubbard parameter",
        "density of states",
        "block cyclic distribution",
        "0.12 eV",
        "Argonne Leadership Computing Facility",
    ],
)
def test_crisp_answers_pass(answer):
    assert not generic_answer(answer) and not clause_answer(answer)


def _cand(**kw):
    base = dict(
        id="c1",
        question="Which group developed MACE?",
        answer="Cambridge Group",
        source_fact_ids=[],
        source_chunk_ids=[],
        strategy="atomic",
        answer_kind="organization",
    )
    return Candidate(**{**base, **kw})


def test_evaluator_form_problems_depend_on_mode(knowledge_base):
    docs, chunks, store = knowledge_base
    rag = Evaluator(mock_config(), ModelRegistry(mock_config()), store, chunks)
    trivia_cfg = mock_config(mode="trivia")
    trivia = Evaluator(trivia_cfg, ModelRegistry(trivia_cfg), store, chunks)
    local = _cand(question="What was the simulation temperature in Run 2?", answer="450 K")
    assert rag.form_problems(local) == []
    assert "local_label" in trivia.form_problems(local)
    assert "source_reference" in rag.form_problems(_cand(question="What did the authors use?"))
    assert "generic_answer" in rag.form_problems(_cand(answer="larger matrix"))
    clause = _cand(answer="only at the beginning of the SCF cycles")
    assert "clause_answer" in trivia.form_problems(clause) and not rag.form_problems(clause)
    assert "answer_kind_other" in trivia.form_problems(_cand(answer_kind="other"))
    assert "answer_not_in_evidence" in trivia.form_problems(_cand())  # no fact backs it
    f = next(x for x in store if not x.object.is_numeric)
    grounded = _cand(answer=f.object.display(), source_fact_ids=[f.id], answer_kind="name")
    assert trivia.form_problems(grounded) == []
    off = _cand(answer="Makefiles", source_fact_ids=[f.id], answer_kind="name")
    assert "answer_not_in_evidence" in trivia.form_problems(off)
    assert "answer_not_in_evidence" not in rag.form_problems(off)


# ------------------------------------------------------------------ config
def test_trivia_mode_tightens_defaults_unless_explicit():
    cfg = Config.model_validate({"mode": "trivia"})
    assert cfg.generation.max_answer_words == cfg.trivia.max_answer_words == 4
    assert cfg.extraction.max_facts_per_chunk == 8
    assert cfg.generation.max_statement_words == 25
    assert cfg.score_trivia_facts
    explicit = Config.model_validate(
        {"mode": "trivia", "generation": {"max_answer_words": 7}, "trivia": {"score_facts": False}}
    )
    assert explicit.generation.max_answer_words == 7 and not explicit.score_trivia_facts
    rag = Config.model_validate({})
    assert rag.generation.max_answer_words == 8 and not rag.score_trivia_facts


# ------------------------------------------------------------------ trivia scoring
def _fresh_store(store: FactStore) -> FactStore:
    return FactStore(f.model_copy(deep=True) for f in store)


async def test_score_trivia_sets_scores_and_briefs(knowledge_base, models):
    docs, chunks, store = knowledge_base
    store = _fresh_store(store)
    briefs = await score_trivia(models["trivia_scorer"], store, chunks, Config().trivia)
    assert set(briefs) == set(store.documents())
    assert all(b.summary and b.nuggets for b in briefs.values())
    assert all(f.trivia_score is not None and 0 <= f.trivia_score <= 1 for f in store)
    routine = [f for f in store if "run" in f.subject.lower()]
    assert routine and all(f.trivia_score <= 0.1 for f in routine)


def test_fact_weight_prefers_high_trivia_scores(knowledge_base, models):
    docs, chunks, store = knowledge_base
    store = _fresh_store(store)
    cfg = mock_config(mode="trivia")
    ctx = GenerationContext(
        cfg, store, build_graph(store), {}, models["question_generator"], random.Random(0)
    )
    a, b = store.all()[:2]
    a.trivia_score, b.trivia_score = 0.9, 0.1
    a.quality = b.quality = None
    a.confidence = b.confidence = 0.9
    assert ctx.fact_weight(a) > 5 * ctx.fact_weight(b)


def test_trivia_context_includes_passage_and_brief(knowledge_base, models):
    from trivium.facts.trivia import DocumentBrief

    docs, chunks, store = knowledge_base
    cfg = mock_config(mode="trivia")
    f = store.all()[0]
    ctx = GenerationContext(
        cfg,
        store,
        build_graph(store),
        {c.id: c for c in chunks},
        models["question_generator"],
        random.Random(0),
        briefs={f.document_id: DocumentBrief(summary="S", nuggets=["N"])},
    )
    tc = ctx.trivia_context([f])
    assert tc["passage"] and tc["document_brief"]["summary"] == "S"
    rag_ctx = GenerationContext(
        mock_config(), store, build_graph(store), {}, models["question_generator"], random.Random(0)
    )
    assert rag_ctx.trivia_context([f]) == {}


# ------------------------------------------------------------------ multiple choice
def _mc_ctx(ctx):
    return MultipleChoiceConverter(ctx)


def test_in_document_pool_is_same_role_and_unit(ctx):
    mae = next(f for f in ctx.store if f.predicate == "mean_absolute_error" and f.subject == "MACE")
    cand = _cand(answer="0.12 eV", source_fact_ids=[mae.id], answer_kind="number")
    pool = _mc_ctx(ctx)._in_document(cand)
    assert set(pool) == {"0.21 eV", "0.35 eV"}  # no training times, no Angstrom values
    founded = next(
        f for f in ctx.store if f.predicate == "founded_in" and f.subject == "Los Alamos Lab"
    )
    years = _mc_ctx(ctx)._in_document(_cand(answer="1943", source_fact_ids=[founded.id]))
    assert "1209" not in years and "1932" in years  # 734 years away is not a confuser


def test_subject_answers_get_competing_subjects(ctx):
    mae = next(f for f in ctx.store if f.predicate == "mean_absolute_error" and f.subject == "MACE")
    cand = _cand(answer="MACE", source_fact_ids=[mae.id], strategy="comparison")
    assert set(_mc_ctx(ctx)._in_document(cand)) == {"XTB", "ANI"}


def test_near_miss_numbers_keep_unit_and_precision(ctx):
    vals = _mc_ctx(ctx)._near_miss(_cand(answer="0.12 eV"))
    assert vals and all(v.endswith(" eV") and len(v.split()[0]) == 4 for v in vals)
    years = _mc_ctx(ctx)._near_miss(_cand(answer="2021"))
    assert years and all(abs(int(y) - 2021) <= 5 for y in years)
    assert _mc_ctx(ctx)._near_miss(_cand(answer="Cambridge Group")) == []


def test_misconceptions_come_from_wrong_closed_book_answers(ctx):
    cand = _cand(answer="Cambridge Group")
    pool = _mc_ctx(ctx)._misconceptions(
        cand, ["Cambridge Group", "unknown", "Oxford Group", "DeepMind"]
    )
    assert pool == ["Oxford Group", "DeepMind"]


def test_stands_out_detects_format_cues():
    assert _stands_out("SIESTA with default solver (ScaLAPACK)", ["ELPA", "MUMPS", "ARPACK"])
    assert _stands_out("42", ["Paris", "London", "Rome"])
    assert _stands_out("ScaLAPACK (default)", ["ELPA", "MUMPS", "ARPACK"])
    assert not _stands_out("MACE", ["XTB", "ANI", "SchNet"])
    assert not _stands_out("0.12 eV", ["0.21 eV", "0.35 eV", "0.15 eV"])


def test_pick_diversifies_kinds():
    valid = [
        DistractorProposal(text=t, kind=k)
        for t, k in [("a", "near_miss"), ("b", "near_miss"), ("c", "near_miss"), ("d", "other")]
    ]
    kinds = [p.kind for p in _pick("x", valid, 3)]
    assert kinds.count("near_miss") == 2 and "other" in kinds


async def test_convert_records_kinds_and_rejects_placeholders(ctx, models):
    mae = next(f for f in ctx.store if f.predicate == "mean_absolute_error" and f.subject == "MACE")
    cand = _cand(
        question="Which model has the lowest mean absolute error?",
        answer="MACE",
        source_fact_ids=[mae.id],
        source_chunk_ids=[mae.chunk_id],
        strategy="comparison",
    )
    conv = MultipleChoiceConverter(ctx, probe=models["closed_book"])
    mc = await conv.convert(cand, [mae.evidence], closed_book_answers=["Model A", "SchNet"])
    assert mc is not None and mc.answer_type == "choice" and len(mc.options) == 4
    kinds = mc.derivation["distractor_kinds"]
    assert {"XTB", "ANI"} <= set(kinds) and kinds["XTB"] == "in_document"
    assert "Model A" not in mc.options  # placeholder-like names are dropped
    assert mc.derivation["options_only_hits"] in (0, 1)


async def test_options_only_probe_rejects_guessable_sets(ctx):
    class Oracle:  # always finds the correct option
        async def generate(self, messages, schema, **kw):
            from trivium.prompts import parse_input

            opts = parse_input(messages)["options"]
            letter = next(o[0] for o in opts if o[3:] == "MACE")

            class R:
                parsed = schema(answer=letter)

            return R()

    mae = next(f for f in ctx.store if f.predicate == "mean_absolute_error" and f.subject == "MACE")
    cand = _cand(answer="MACE", source_fact_ids=[mae.id], strategy="comparison")
    conv = MultipleChoiceConverter(ctx, probe=Oracle())
    assert await conv.convert(cand, [mae.evidence]) is None
    assert conv.last_failure == "options_only_guessable"


def test_option_picks_counts_letters_and_text():
    opts = ["MACE", "XTB", "ANI", "SchNet"]
    assert option_picks(opts, ["A", "B)", "ani", "A"]) == {
        "MACE": 2,
        "XTB": 1,
        "ANI": 1,
        "SchNet": 0,
    }


# ------------------------------------------------------------------ true / false
@pytest.mark.parametrize(
    "kind",
    ["subject_swap", "comparison_flip", "llm_edit", "numeric_perturbation", "sibling_object_swap"],
)
async def test_each_falsification_is_contradicted(ctx, kind):
    gen = TrueFalseGenerator(ctx)
    found = []
    for f in ctx.store.all():
        claim = gen._false_claim(f, kind=kind)
        if claim is None:
            continue
        c = await gen._realise(claim)
        if c is not None:
            found.append(c)
    assert found, kind
    for c in found:
        assert c.answer == "False" and c.question.startswith("True or False:")
        assert c.derivation["perturbation"] == kind
        first = ctx.store.get(c.source_fact_ids[0])
        assert norm(c.derivation["asserted_object"]) != norm(first.object.display())


async def test_false_claims_sample_only_applicable_kinds(ctx):
    gen = TrueFalseGenerator(ctx)
    kinds: set[str] = set()
    for _ in range(8):
        cands = await gen.generate(6)
        kinds |= {c.derivation["perturbation"] for c in cands if c.answer == "False"}
    assert len(kinds) >= 2  # not just numeric perturbation


def test_comparison_statements_can_be_true(ctx):
    ctx.rng.random = lambda: 0.1  # below the 0.3 comparison share
    mae = next(f for f in ctx.store if f.predicate == "mean_absolute_error" and f.subject == "MACE")
    claim = TrueFalseGenerator(ctx)._true_claim(mae)
    assert claim.kind == "comparison_true" and claim.truth
    assert claim.asserted.startswith("lower than")  # MACE has the lowest MAE


async def test_long_statements_are_dropped(ctx):
    ctx.config.generation.max_statement_words, old = 3, ctx.config.generation.max_statement_words
    try:
        cands = await TrueFalseGenerator(ctx).generate(6)
        assert cands == []
    finally:
        ctx.config.generation.max_statement_words = old


# ------------------------------------------------------------------ ranking
def _ec(cid, interest, question):
    cand = Candidate(
        id=cid,
        question=question,
        answer="a",
        source_fact_ids=[],
        source_chunk_ids=[],
        strategy="atomic",
    )
    ev = Evaluation(
        difficulty=Difficulty.medium,
        interestingness=interest,
        specificity=0.8,
        clarity=0.9,
        accepted=True,
    )
    return EvaluatedCandidate(candidate=cand, evaluation=ev, evidence=[])


async def test_tournament_win_rates_drive_interest(models):
    pool = [_ec(f"c{i}", 0.7, f"Question number {i}?") for i in range(6)]
    rates = await selection.tournament(models["judge"], pool, rounds=3, rng=random.Random(0))
    assert set(rates) == {ec.candidate.id for ec in pool}
    assert all(0 <= r <= 1 for r in rates.values()) and len(set(rates.values())) > 1
    ranked = selection.rank(pool, "interest", pairwise=rates)
    assert ranked[0].evaluation.pairwise_win_rate == max(rates.values())


async def test_end_to_end_trivia_mode_best(tmp_path):
    cfg = mock_config(
        mode="trivia",
        output=str(tmp_path / "q.json"),
        generation={
            "num_questions": 10,
            "select_top": 3,
            "question_types": ["short_answer", "true_false", "multiple_choice"],
            "seed": 3,
        },
    )
    ds, run = await generate_dataset(cfg, write=False)
    assert len(ds.questions) == 3
    assert all(q.evaluation.pairwise_win_rate is not None for q in ds.questions)
    assert any("brief" in s for s in ds.dataset.sources)
    assert all(f.trivia_score is not None for f in run.store)
    for q in ds.questions:
        assert not local_label(q.question) and not source_reference(q.question)
        if q.type == "short_answer":
            assert len(q.answer.split()) <= 4
