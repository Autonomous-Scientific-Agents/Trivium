import json

import pytest
from conftest import DATA, mock_config

from trivium.cli import main
from trivium.config import Config, Difficulty
from trivium.evaluation.evaluator import EvaluatedCandidate, Evaluation
from trivium.generation.base import Candidate
from trivium.orchestration import selection
from trivium.orchestration.controller import generate_dataset


def _ec(cid, difficulty, tiers, cb_cc, depth, interest, quality=0.9, spec=0.8, clarity=0.9):
    cand = Candidate(
        id=cid,
        question="q?",
        answer="a",
        source_fact_ids=["f"],
        source_chunk_ids=["c"],
        strategy="atomic",
        reasoning_depth=depth,
    )
    ev = Evaluation(
        difficulty=difficulty,
        tier_accuracy=tiers,
        closed_book_chance_corrected=cb_cc,
        closed_book_accuracy=cb_cc,
        interestingness=interest,
        specificity=spec,
        clarity=clarity,
        overall_quality=quality,
        accepted=True,
    )
    return EvaluatedCandidate(candidate=cand, evaluation=ev, evidence=[])


def test_scores_are_bounded_and_ordered():
    easy = _ec("e", Difficulty.easy, {"small": 1.0, "medium": 1.0}, 1.0, 1, 0.2)
    hard = _ec("h", Difficulty.hard, {"small": 0.0, "medium": 0.5}, 0.0, 2, 0.6)
    extreme = _ec("x", Difficulty.extreme, {"small": 0.0, "medium": 0.0}, 0.0, 3, 0.3)
    for ec in (easy, hard, extreme):
        assert 0 <= selection.challenge_score(ec) <= 1 and 0 <= selection.interest_score(ec) <= 1
    assert (
        selection.challenge_score(easy)
        < selection.challenge_score(hard)
        < selection.challenge_score(extreme)
    )
    assert selection.interest_score(hard) > selection.interest_score(extreme)
    ranked = selection.rank([easy, extreme, hard], "challenge")
    assert [ec.candidate.id for ec in ranked] == ["x", "h", "e"]
    assert [ec.evaluation.selection_rank for ec in ranked] == [1, 2, 3]
    # trivia mixes both: the interesting hard one can beat the dull extreme one
    boring_extreme = _ec(
        "bx", Difficulty.extreme, {"small": 0.0, "medium": 0.0}, 0.0, 3, 0.0, spec=0.2, clarity=0.3
    )
    fun_hard = _ec("fh", Difficulty.hard, {"small": 0.0, "medium": 0.5}, 0.0, 2, 1.0)
    assert selection.rank([boring_extreme, fun_hard], "trivia")[0].candidate.id == "fh"
    with pytest.raises(ValueError):
        selection.selection_score(easy, "bogus")


def test_no_tiers_falls_back_to_closed_book():
    a = _ec("a", Difficulty.hard, {}, 0.0, 1, 0.5)
    b = _ec("b", Difficulty.hard, {}, 1.0, 1, 0.5)
    assert selection.challenge_score(a) > selection.challenge_score(b)


def test_config_derives_pool_and_difficulty():
    g = Config.model_validate({"generation": {"select_top": 1}}).generation
    assert g.num_questions == 10 and g.prefer_difficulty == "hard"
    g = Config.model_validate({"generation": {"select_top": 3, "num_questions": 40}}).generation
    assert g.num_questions == 40
    with pytest.raises(ValueError):
        Config.model_validate({"generation": {"select_top": 5, "num_questions": 3}})
    assert (
        Config.model_validate(
            {"mode": "trivia", "generation": {"select_top": 1}}
        ).effective_select_by
        == "trivia"
    )
    assert (
        Config.model_validate({"generation": {"select_top": 1}}).effective_select_by == "challenge"
    )


async def test_best_of_selection_end_to_end(tmp_path):
    cfg = mock_config(
        mode="trivia",
        output=str(tmp_path / "q.json"),
        jsonl_dir=str(tmp_path / "b"),
        generation={
            "select_top": 2,
            "seed": 3,
            "question_types": ["short_answer", "multiple_choice"],
        },
    )
    ds, run = await generate_dataset(cfg)
    assert len(ds.questions) == 2 and run.target == 16 and run.selected_by == "trivia"
    assert ds.dataset.selected_top == 2 and ds.dataset.selected_by == "trivia"
    scores = [q.evaluation.selection_score for q in ds.questions]
    assert scores == sorted(scores, reverse=True) and [
        q.evaluation.selection_rank for q in ds.questions
    ] == [1, 2]
    rejected = [
        json.loads(line) for line in (tmp_path / "b" / "rejected.jsonl").read_text().splitlines()
    ]
    not_selected = [r for r in rejected if "not_selected" in r["evaluation"]["rejection_reasons"]]
    assert len(not_selected) == 14
    assert all(r["evaluation"]["selection_score"] <= scores[-1] for r in not_selected)
    # hard steering from round one: the feedback carried the target difficulty
    assert all(q.generation.target_difficulty == "hard" for q in ds.questions)


def test_cli_best_flag(tmp_path, capsys):
    out = tmp_path / "best.json"
    rc = main(
        [
            "generate",
            "--input",
            str(DATA),
            "--provider",
            "mock",
            "--best",
            "1",
            "--mode",
            "trivia",
            "-o",
            str(out),
            "--no-cache",
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["accepted"] == 1 and summary["pool"] == 10 and summary["selected_by"] == "trivia"
