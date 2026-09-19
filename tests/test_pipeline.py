import json

from conftest import DATA, mock_config

from trivium.cli import main
from trivium.config import Config, QuestionType
from trivium.orchestration.controller import generate_dataset
from trivium.orchestration.stopping import LoopState, estimate_target, should_stop
from trivium.orchestration.strategy import StrategySelector
from trivium.output.schema import Dataset


async def test_end_to_end_mock(tmp_path):
    cfg = mock_config(
        output=str(tmp_path / "q.json"),
        jsonl_dir=str(tmp_path / "bundle"),
        generation={
            "num_questions": 20,
            "question_types": ["short_answer", "true_false", "multiple_choice"],
            "multi_hop_ratio": 0.2,
            "cross_document_ratio": 0.1,
            "seed": 5,
        },
    )
    dataset, run = await generate_dataset(cfg)
    assert run.stop_reason == "quota_reached"
    assert len(dataset.questions) == 20
    assert run.n_candidates > 20  # more candidates than accepted (design §2.1)
    m = dataset.dataset_metrics
    assert set(m["type_distribution"]) >= {"short_answer", "multiple_choice"}
    assert m["candidate_acceptance_rate"] < 1.0
    assert m["average_document_dependence"] > 0.8
    assert dataset.coverage["section_coverage"] > 0.5
    assert dataset.strategy_stats["atomic"]["generated"] > 0
    ids = [q.id for q in dataset.questions]
    assert ids == sorted(ids) and ids[0] == "q_0001"
    for q in dataset.questions:
        assert q.evidence and all(e.text for e in q.evidence)
        assert q.evaluation.accepted and not q.evaluation.rejection_reasons
        assert q.difficulty in ("easy", "medium", "hard")
        assert q.provenance["generation_model"].startswith("mock:")
        if q.type == "multiple_choice":
            assert q.options and len(q.options) == 4 and q.answer in "ABCD"
            assert q.options["ABCD".index(q.answer)] in q.accepted_answers
    # files
    data = json.loads((tmp_path / "q.json").read_text())
    Dataset.model_validate(data)
    for name in (
        "metadata.json",
        "questions.jsonl",
        "facts.jsonl",
        "documents.jsonl",
        "chunks.jsonl",
        "rejected.jsonl",
    ):
        assert (tmp_path / "bundle" / name).exists(), name
    rejected = [
        json.loads(line)
        for line in (tmp_path / "bundle" / "rejected.jsonl").read_text().splitlines()
    ]
    assert rejected and all(r["evaluation"]["rejection_reasons"] for r in rejected)


async def test_run_is_reproducible_with_seed(tmp_path):
    outs = []
    for i in range(2):
        cfg = mock_config(
            output=str(tmp_path / f"q{i}.json"), generation={"num_questions": 12, "seed": 42}
        )
        ds, _ = await generate_dataset(cfg, write=False)
        outs.append([(q.question, q.answer) for q in ds.questions])
    assert outs[0] == outs[1]


async def test_difficulty_quotas_and_auto_target(tmp_path):
    cfg = mock_config(
        output=str(tmp_path / "q.json"), generation={"medium": 4, "hard": 3, "seed": 2}
    )
    assert cfg.generation.num_questions == 7
    ds, run = await generate_dataset(cfg, write=False)
    assert run.stop_reason in ("quota_reached", "difficulty_quotas_reached")
    dist = ds.dataset_metrics["difficulty_distribution"]
    assert dist.get("hard", 0) >= 3 and dist.get("medium", 0) >= 4

    auto = mock_config(output=str(tmp_path / "a.json"), generation={"max_questions": 15, "seed": 2})
    assert auto.generation.num_questions is None
    ds, run = await generate_dataset(auto, write=False)
    assert 1 <= run.target <= 15 and len(ds.questions) <= run.target


def test_stopping_and_target_heuristic():
    cfg = Config().generation
    assert should_stop(LoopState(accepted=cfg.num_questions or 10), cfg, 10) == "quota_reached"
    assert should_stop(LoopState(rounds_without_acceptance=cfg.patience), cfg, 10) == "exhaustion"
    assert should_stop(LoopState(low_novelty_rounds=2), cfg, 10) == "excessive_redundancy"
    assert should_stop(LoopState(), cfg, 10) is None
    assert estimate_target(100, 50, 30, 500) == 53
    assert estimate_target(1000, 0, 0, 100) == 100


def test_strategy_allocation_sums_to_batch():
    import random

    from trivium.orchestration.coverage import CoverageReport

    sel = StrategySelector(
        Config().generation,
        {"atomic": True, "comparison": True, "multi_hop": True, "derived": False},
    )
    w = sel.weights(CoverageReport(), "hard", 10)
    assert "derived" not in w and abs(sum(w.values()) - 1) < 1e-9
    assert w["multi_hop"] > w["atomic"]  # hard target favours multi-hop
    alloc = sel.allocate(8, w, random.Random(0))
    assert sum(alloc.values()) == 8


def test_config_cli_overrides_and_yaml(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text(
        "mode: trivia\ngeneration:\n  max_answer_words: 5\nmodels:\n  default: {provider: mock}\n  roles:\n    difficulty_small: {model: tiny}\n"
    )
    cfg = Config.load(y, generation={"num_questions": 3}, cache={"enabled": False})
    assert (
        cfg.mode.value == "trivia"
        and cfg.generation.max_answer_words == 5
        and cfg.generation.num_questions == 3
    )
    assert (
        cfg.models.spec_for("difficulty_small").model == "tiny"
        and cfg.models.spec_for("difficulty_small").provider == "mock"
    )
    assert cfg.models.spec_for("difficulty_medium") is None
    assert cfg.quality_weights["interestingness"] > Config().quality_weights["interestingness"]


def test_cli_generate_and_stats(tmp_path, capsys):
    out = tmp_path / "cli.json"
    rc = main(
        [
            "generate",
            "--input",
            str(DATA),
            "--provider",
            "mock",
            "-n",
            "6",
            "-o",
            str(out),
            "--no-cache",
            "--seed",
            "1",
            "--question-types",
            "short_answer",
        ]
    )
    assert rc == 0 and out.exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary["accepted"] == 6 and summary["stop_reason"] == "quota_reached"
    data = json.loads(out.read_text())
    assert all(q["type"] == QuestionType.short_answer.value for q in data["questions"])
    assert main(["stats", str(out)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["dataset_metrics"]["num_questions"] == 6


def test_cli_extract(tmp_path, capsys):
    out = tmp_path / "facts.jsonl"
    assert (
        main(["extract", "--input", str(DATA), "--provider", "mock", "-o", str(out), "--no-cache"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["facts"] >= 25
    assert len(out.read_text().splitlines()) >= 25
