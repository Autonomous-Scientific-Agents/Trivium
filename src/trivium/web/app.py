"""FastAPI app: upload sources, run generation, review questions and record feedback.

Start it with ``trivium serve`` (see ``trivium serve --help``). There is no authentication: bind
it to localhost (the default) or put it behind something that authenticates.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from trivium.config import Config
from trivium.ingestion.loader import SUPPORTED
from trivium.web.jobs import JobQueue, RunSettings, build_config
from trivium.web.store import Workspace, display_answer, now, safe_filename

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class FeedbackIn(BaseModel):
    rating: str | None = None
    comment: str = ""
    reviewer: str | None = None


def _tail(path: Path, n: int = 40) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


def create_app(
    workspace: str | Path = "runs",
    *,
    base_config: str | Path | None = None,
    defaults: dict[str, Any] | None = None,
    jobs: JobQueue | None = None,
) -> FastAPI:
    ws = Workspace(workspace)
    jobs = jobs or JobQueue(ws, base_config=base_config, defaults=defaults)
    base = Config.load(base_config)  # config file + TRIVIUM_* environment

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        jobs.start()
        yield
        jobs.stop()

    app = FastAPI(title="trivium review", lifespan=lifespan)
    app.state.workspace = ws
    app.state.jobs = jobs

    def run_dir(run_id: str) -> Path:
        try:
            return ws.run_dir(run_id)
        except KeyError:
            raise HTTPException(404, f"no run {run_id!r}") from None

    def status(d: Path) -> dict[str, Any]:
        job = ws.job(d) or {}
        st = job.get("status") or ("done" if ws.dataset_path(d) else "unknown")
        return {
            "status": st,
            "queue_position": jobs.position(d.name) if st == "queued" else None,
            "error": job.get("error"),
            "summary": job.get("summary"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "log": _tail(d / "generate.log"),
        }

    # ------------------------------------------------------------------ API
    @app.get("/api/settings")
    def settings() -> dict[str, Any]:
        spec = base.models.default
        d = defaults or {}
        return {
            "workspace": str(ws.root),
            "config": str(base_config) if base_config else None,
            "provider": d.get("provider") or spec.provider,
            "model": d.get("model") or spec.model,
            "mode": base.mode.value if base_config else "trivia",
            "question_types": [t.value for t in base.generation.question_types]
            if base_config
            else RunSettings().question_types,
            "extensions": sorted(SUPPORTED),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        }

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return ws.list_runs()

    @app.post("/api/runs", status_code=201)
    async def create_run(
        settings: Annotated[str, Form()] = "{}",
        files: Annotated[list[UploadFile] | None, File()] = None,
        text: Annotated[str, Form()] = "",
    ) -> dict[str, Any]:
        try:
            parsed = RunSettings.model_validate(json.loads(settings or "{}"))
        except (ValueError, ValidationError) as exc:
            raise HTTPException(422, f"invalid settings: {exc}") from None
        uploads: list[tuple[str, bytes]] = []
        for f in files or []:
            name = safe_filename(f.filename or "source")
            if Path(name).suffix.lower() not in SUPPORTED:
                raise HTTPException(415, f"unsupported file type: {f.filename}")
            data = await f.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{f.filename} is larger than {MAX_UPLOAD_BYTES} bytes")
            if data:
                uploads.append((name, data))
        if text.strip():
            uploads.append(("pasted_text.txt", text.encode("utf-8")))
        if not uploads:
            raise HTTPException(422, "upload at least one file or paste some text")
        name = parsed.name or Path(uploads[0][0]).stem
        run_id, d = ws.create_run(name)
        saved: list[str] = []
        for fname, data in uploads:
            target, i = d / "sources" / fname, 2
            while target.exists():
                target = d / "sources" / f"{Path(fname).stem}_{i}{Path(fname).suffix}"
                i += 1
            target.write_bytes(data)
            saved.append(target.name)
        parsed.name = name
        try:  # surface config errors now rather than in the background job
            build_config(parsed, d, base_config=base_config, defaults=defaults)
        except Exception as exc:
            ws.save_job(d, {"status": "failed", "error": str(exc), "created_at": now()})
            raise HTTPException(422, f"invalid configuration: {exc}") from None
        ws.save_job(
            d,
            {
                "status": "queued",
                "created_at": now(),
                "settings": parsed.model_dump(),
                "files": saved,
            },
        )
        jobs.submit(run_id)
        return {"id": run_id, **status(d)}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        d = run_dir(run_id)
        job = ws.job(d)
        data = ws.load_dataset(d) or {}
        return {
            **ws.summary(d, job=job, data=data),
            **status(d),
            "settings": (job or {}).get("settings"),
            "files": (job or {}).get("files", []),
            "dataset": {k: v for k, v in data.get("dataset", {}).items() if k != "config"},
            "dataset_metrics": data.get("dataset_metrics"),
            "persona_gaps": (data.get("strategy_stats") or {}).get("persona_gaps", []),
            "questions": data.get("questions", []),
            "feedback": ws.feedback(d),
        }

    @app.get("/api/runs/{run_id}/status")
    def get_status(run_id: str) -> dict[str, Any]:
        return status(run_dir(run_id))

    @app.post("/api/runs/{run_id}/retry")
    def retry(run_id: str) -> dict[str, Any]:
        d = run_dir(run_id)
        job = ws.job(d)
        if not job or not job.get("settings"):
            raise HTTPException(409, "this run was not created from the web UI")
        if job.get("status") in ("queued", "running"):
            raise HTTPException(409, "run is already queued or running")
        ws.archive_feedback(d)  # regenerated questions reuse ids: old ratings must not carry over
        jobs.submit(run_id)
        return status(d)

    @app.put("/api/runs/{run_id}/feedback/{qid}")
    def put_feedback(run_id: str, qid: str, body: FeedbackIn) -> dict[str, Any]:
        d = run_dir(run_id)
        data = ws.load_dataset(d) or {}
        question = next((q for q in data.get("questions", []) if q.get("id") == qid), None)
        if question is None:
            raise HTTPException(404, f"no question {qid!r} in run {run_id!r}")
        try:
            entry = ws.set_feedback(
                d,
                qid,
                rating=body.rating,
                comment=body.comment,
                reviewer=body.reviewer,
                question=question,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return {"id": qid, "feedback": entry}

    @app.get("/api/runs/{run_id}/feedback.jsonl")
    def export_feedback(run_id: str) -> Response:
        """One line per question (rated or not), with the reviewer's rating and comment."""
        d = run_dir(run_id)
        data = ws.load_dataset(d) or {}
        fb = ws.feedback(d)
        lines = []
        for q in data.get("questions", []):
            f = fb.get(q.get("id"), {})
            lines.append(
                json.dumps(
                    {
                        "run": run_id,
                        "id": q.get("id"),
                        "type": q.get("type"),
                        "question": q.get("question"),
                        "options": q.get("options"),
                        "answer": display_answer(q),
                        "difficulty": q.get("difficulty"),
                        "strategy": (q.get("generation") or {}).get("strategy"),
                        "rating": f.get("rating"),
                        "comment": f.get("comment", ""),
                        "reviewer": f.get("reviewer"),
                        "updated_at": f.get("updated_at"),
                    },
                    ensure_ascii=False,
                )
            )
        return PlainTextResponse(
            "\n".join(lines) + ("\n" if lines else ""),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_feedback.jsonl"'},
        )

    @app.get("/api/runs/{run_id}/questions.json")
    def download_questions(run_id: str) -> FileResponse:
        d = run_dir(run_id)
        p = ws.dataset_path(d)
        if p is None:
            raise HTTPException(404, "no dataset yet")
        return FileResponse(p, filename=f"{run_id}_questions.json")

    # ------------------------------------------------------------- frontend
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
