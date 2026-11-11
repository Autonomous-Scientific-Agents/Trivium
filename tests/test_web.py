"""Review web UI: upload → generate (mock backend) → review → feedback → export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from trivium.web.app import create_app  # noqa: E402
from trivium.web.jobs import JobQueue  # noqa: E402
from trivium.web.store import Workspace, display_answer, safe_filename  # noqa: E402

DATA = Path(__file__).parent / "data"


@pytest.fixture
def base_config(tmp_path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        "cache: {enabled: false}\n"
        "log_level: WARNING\n"
        "generation: {seed: 11, batch_size: 6}\n"
        "models:\n  default: {provider: mock, model: mock-large}\n"
        "  embeddings: {provider: hashing}\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def client(tmp_path, base_config):
    app = create_app(tmp_path / "ws", base_config=base_config)
    with TestClient(app) as c:
        yield c


def _upload(client, settings: dict, files=None, text: str = ""):
    files = files if files is not None else [DATA / "datasets.md"]
    return client.post(
        "/api/runs",
        data={"settings": json.dumps(settings), "text": text},
        files=[("files", (p.name, p.read_bytes(), "text/markdown")) for p in files],
    )


def test_upload_generate_review_and_export(client):
    s = client.get("/api/settings").json()
    assert s["provider"] == "mock" and ".pdf" in s["extensions"]

    r = _upload(
        client,
        {
            "name": "Datasets!",
            "mode": "trivia",
            "num_questions": 4,
            "question_types": ["short_answer", "true_false"],
        },
    )
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]
    assert run_id.endswith("datasets")
    client.app.state.jobs.join(timeout=120)

    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "done", run.get("error") or run["log"]
    assert run["files"] == ["datasets.md"]
    assert run["mode"] == "trivia"
    assert 1 <= len(run["questions"]) <= 4
    assert any("generating" in line for line in run["log"])
    q1, *rest = run["questions"]

    fb = client.put(
        f"/api/runs/{run_id}/feedback/{q1['id']}",
        json={"rating": "up", "comment": "  nice one ", "reviewer": "mk"},
    )
    assert fb.status_code == 200
    assert fb.json()["feedback"]["comment"] == "nice one"
    if rest:
        client.put(f"/api/runs/{run_id}/feedback/{rest[0]['id']}", json={"rating": "down"})

    listed = {x["id"]: x for x in client.get("/api/runs").json()}[run_id]
    assert listed["num_up"] == 1 and listed["num_down"] == (1 if rest else 0)
    assert listed["name"] == "Datasets!"

    lines = client.get(f"/api/runs/{run_id}/feedback.jsonl").text.splitlines()
    rows = {row["id"]: row for row in map(json.loads, lines)}
    assert len(rows) == len(run["questions"])
    assert rows[q1["id"]]["rating"] == "up" and rows[q1["id"]]["reviewer"] == "mk"
    assert rows[q1["id"]]["question"] == q1["question"]

    # clearing both rating and comment removes the entry
    client.put(f"/api/runs/{run_id}/feedback/{q1['id']}", json={"rating": None, "comment": ""})
    assert q1["id"] not in client.get(f"/api/runs/{run_id}").json()["feedback"]

    assert client.get(f"/api/runs/{run_id}/questions.json").json()["questions"]


def test_pasted_text_and_validation(client):
    r = client.post(
        "/api/runs",
        data={"settings": json.dumps({"name": "notes"}), "text": (DATA / "models.md").read_text()},
    )
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]
    client.app.state.jobs.join(timeout=120)
    assert client.get(f"/api/runs/{run_id}").json()["files"] == ["pasted_text.txt"]

    assert client.post("/api/runs", data={"settings": "{}"}).status_code == 422  # nothing to use
    bad = client.post(
        "/api/runs",
        data={"settings": "{}"},
        files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))],
    )
    assert bad.status_code == 415
    assert _upload(client, {"question_types": ["essay"]}).status_code == 422
    assert _upload(client, {"mode": "quiz"}).status_code == 422

    run = client.get(f"/api/runs/{run_id}").json()
    qid = run["questions"][0]["id"]
    assert (
        client.put(f"/api/runs/{run_id}/feedback/{qid}", json={"rating": "meh"}).status_code == 422
    )
    assert (
        client.put(f"/api/runs/{run_id}/feedback/q_nope", json={"rating": "up"}).status_code == 404
    )
    for rid in ("..", "../etc", "%2e%2e", "nope"):
        assert client.get(f"/api/runs/{rid}").status_code == 404


def test_existing_cli_runs_are_listed(tmp_path, base_config):
    ws = tmp_path / "ws"
    (ws / "old").mkdir(parents=True)
    dataset = {
        "dataset": {
            "created_at": "2026-01-01T00:00:00",
            "mode": "rag",
            "sources": [{"title": "Paper"}],
        },
        "questions": [
            {
                "id": "q_0001",
                "question": "Q?",
                "type": "multiple_choice",
                "options": ["x", "y"],
                "answer": "B",
            }
        ],
    }
    (ws / "old" / "q.json").write_text(json.dumps(dataset))
    (ws / "old" / "metadata.json").write_text("{}")
    (ws / "empty").mkdir()
    with TestClient(create_app(ws, base_config=base_config)) as c:
        runs = c.get("/api/runs").json()
        assert [r["id"] for r in runs] == ["old"]
        assert runs[0]["status"] == "done" and runs[0]["sources"] == ["Paper"]
        c.put("/api/runs/old/feedback/q_0001", json={"rating": "down", "comment": "too easy"})
        row = json.loads(c.get("/api/runs/old/feedback.jsonl").text)
        assert row["answer"] == "B) y" and row["comment"] == "too easy"
        assert c.post("/api/runs/old/retry").status_code == 409


def test_failed_job_and_interrupted_on_restart(tmp_path, base_config):
    ws = Workspace(tmp_path / "ws")

    def boom(cfg):
        raise RuntimeError("no validated facts")

    jobs = JobQueue(ws, base_config=base_config, runner=boom)
    with TestClient(create_app(ws.root, base_config=base_config, jobs=jobs)) as c:
        run_id = _upload(c, {}).json()["id"]
        jobs.join(timeout=10)
        run = c.get(f"/api/runs/{run_id}").json()
        assert run["status"] == "failed" and "no validated facts" in run["error"]

    # a job still "running" when the server stopped is reported as interrupted by the next one
    ws.update_job(ws.run_dir(run_id), status="running")
    JobQueue(ws, base_config=base_config, runner=boom)
    assert ws.job(ws.run_dir(run_id))["status"] == "interrupted"


def test_helpers():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("My Paper (final).PDF") == "My_Paper_final.pdf"
    assert safe_filename("..\\x.md") == "x.md"
    assert (
        display_answer({"type": "multiple_choice", "options": ["a", "b"], "answer": "A"}) == "A) a"
    )
    assert display_answer({"type": "true_false", "answer": "True"}) == "True"


def test_static_frontend_served(client):
    assert "trivium" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_pick_port_skips_busy_ports():
    import socket

    from trivium.cli import pick_port

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        assert pick_port("127.0.0.1", None, start=taken) != taken
        with pytest.raises(SystemExit, match="already in use"):
            pick_port("127.0.0.1", taken)
