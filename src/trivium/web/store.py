"""On-disk layout of the review workspace.

A *run* is a directory under the workspace. Runs created from the web UI look like::

    <workspace>/<run_id>/
        job.json          # settings and status of the generation job
        sources/          # uploaded files
        questions.json    # dataset written by generate_dataset
        bundle/           # JSONL bundle (facts, documents, rejected candidates)
        generate.log      # log of the job
        feedback.json     # reviewer ratings and comments, keyed by question id

Directories written by the CLI (``trivium generate -o runs/x/q.json``) are picked up too: any
JSON file at the top of a run directory with ``dataset`` and ``questions`` keys is a dataset.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_FEEDBACK = "feedback.json"
_JOB = "job.json"
_SKIP_JSON = {_FEEDBACK, _JOB, "metadata.json"}
RATINGS = ("up", "down")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def safe_filename(name: str) -> str:
    """Basename only, with anything outside ``[A-Za-z0-9._-]`` replaced."""
    base = Path(name.replace("\\", "/")).name
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "source"
    ext = re.sub(r"[^A-Za-z0-9]+", "", ext).lower()
    return f"{stem[:80]}.{ext}" if ext else stem[:80]


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ runs
    def run_dir(self, run_id: str) -> Path:
        if not RUN_ID.match(run_id):
            raise KeyError(run_id)
        path = (self.root / run_id).resolve()
        if path.parent != self.root or not path.is_dir():
            raise KeyError(run_id)
        return path

    def new_run_id(self, name: str | None = None) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9]+", "-", name or "").strip("-").lower()[:40]
        base = f"{stamp}-{slug}" if slug else stamp
        run_id, i = base, 2
        while (self.root / run_id).exists():
            run_id, i = f"{base}-{i}", i + 1
        return run_id

    def create_run(self, name: str | None = None) -> tuple[str, Path]:
        with self._lock:
            run_id = self.new_run_id(name)
            path = self.root / run_id
            (path / "sources").mkdir(parents=True)
        return run_id, path

    def dataset_path(self, run_dir: Path) -> Path | None:
        preferred = run_dir / "questions.json"
        candidates = [preferred] if preferred.is_file() else []
        candidates += sorted(
            p for p in run_dir.glob("*.json") if p != preferred and p.name not in _SKIP_JSON
        )
        for p in candidates:
            data = read_json(p)
            if isinstance(data, dict) and "dataset" in data and "questions" in data:
                return p
        return None

    def load_dataset(self, run_dir: Path) -> dict[str, Any] | None:
        p = self.dataset_path(run_dir)
        return read_json(p) if p else None

    def job(self, run_dir: Path) -> dict[str, Any] | None:
        return read_json(run_dir / _JOB)

    def save_job(self, run_dir: Path, job: dict[str, Any]) -> None:
        with self._lock:
            write_json_atomic(run_dir / _JOB, job)

    def update_job(self, run_dir: Path, **fields: Any) -> dict[str, Any]:
        with self._lock:
            job = read_json(run_dir / _JOB) or {}
            job.update(fields)
            write_json_atomic(run_dir / _JOB, job)
            return job

    def list_runs(self) -> list[dict[str, Any]]:
        out = []
        for d in self.root.iterdir():
            if not d.is_dir() or not RUN_ID.match(d.name):
                continue
            job = self.job(d)
            data = self.load_dataset(d)
            if job is None and data is None:
                continue
            out.append(self.summary(d, job=job, data=data))
        out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return out

    def summary(
        self,
        run_dir: Path,
        *,
        job: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        info = (data or {}).get("dataset", {})
        questions = (data or {}).get("questions", [])
        fb = self.feedback(run_dir)
        ids = {q.get("id") for q in questions}
        ratings = [v.get("rating") for k, v in fb.items() if k in ids]
        settings = (job or {}).get("settings", {})
        return {
            "id": run_dir.name,
            "name": settings.get("name") or run_dir.name,
            "created_at": (job or {}).get("created_at") or info.get("created_at"),
            "status": (job or {}).get("status") or ("done" if data else "unknown"),
            "mode": info.get("mode") or settings.get("mode"),
            "sources": [s.get("title") or s.get("document_id") for s in info.get("sources", [])]
            or (job or {}).get("files", []),
            "num_questions": len(questions),
            "num_up": ratings.count("up"),
            "num_down": ratings.count("down"),
            "num_commented": sum(1 for k, v in fb.items() if k in ids and v.get("comment")),
        }

    # -------------------------------------------------------------- feedback
    def feedback(self, run_dir: Path) -> dict[str, dict[str, Any]]:
        data = read_json(run_dir / _FEEDBACK)
        return data.get("items", {}) if isinstance(data, dict) else {}

    def set_feedback(
        self,
        run_dir: Path,
        qid: str,
        *,
        rating: str | None,
        comment: str,
        reviewer: str | None = None,
        question: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Store (or clear, when there is neither a rating nor a comment) one question's feedback.

        A snapshot of the question and answer is kept with the feedback so it stays meaningful
        if the dataset file is regenerated.
        """
        if rating is not None and rating not in RATINGS:
            raise ValueError(f"rating must be one of {RATINGS} or null")
        comment = comment.strip()
        with self._lock:
            data = read_json(run_dir / _FEEDBACK)
            if not isinstance(data, dict):
                data = {"version": 1, "items": {}}
            items = data.setdefault("items", {})
            if rating is None and not comment:
                items.pop(qid, None)
                entry = None
            else:
                entry = {"rating": rating, "comment": comment, "updated_at": now()}
                if reviewer:
                    entry["reviewer"] = reviewer
                if question:
                    entry["question"] = question.get("question")
                    entry["answer"] = display_answer(question)
                    entry["type"] = question.get("type")
                items[qid] = entry
            write_json_atomic(run_dir / _FEEDBACK, data)
        return entry


def display_answer(q: dict[str, Any]) -> str:
    """The answer as a reader sees it: the option text for multiple choice ("B) 16000")."""
    ans = str(q.get("answer", ""))
    opts = q.get("options") or []
    if q.get("type") == "multiple_choice" and len(ans) == 1 and ans.isalpha():
        i = ord(ans.upper()) - ord("A")
        if 0 <= i < len(opts):
            return f"{ans.upper()}) {opts[i]}"
    return ans
