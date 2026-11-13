"""Background generation jobs for the web UI.

Jobs run one at a time on a worker thread (LLM endpoints are rate limited, and the response cache
is shared). Each job calls :func:`trivium.orchestration.controller.generate_dataset` in its own
event loop; its ``trivium`` log records are copied to ``<run>/generate.log`` so the UI can show
progress.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from trivium.config import Config, QuestionType
from trivium.web.store import Workspace, now

log = logging.getLogger(__name__)


class RunSettings(BaseModel):
    """What the "new run" form submits. Everything else comes from the server's config file."""

    name: str = ""
    mode: str = "trivia"
    num_questions: int | None = Field(default=None, ge=1, le=500)
    best: int | None = Field(default=None, ge=1, le=200)
    question_types: list[str] = Field(
        default_factory=lambda: ["short_answer", "true_false", "multiple_choice"]
    )
    provider: str | None = None
    model: str | None = None
    max_chunks: int | None = Field(default=None, ge=1)
    seed: int | None = None
    personas: list[str] = Field(default_factory=list)

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in ("trivia", "rag"):
            raise ValueError("mode must be trivia or rag")
        return v

    @field_validator("question_types")
    @classmethod
    def _types(cls, v: list[str]) -> list[str]:
        allowed = {t.value for t in QuestionType}
        bad = [t for t in v if t not in allowed]
        if bad or not v:
            raise ValueError(f"question_types must be a non-empty subset of {sorted(allowed)}")
        return v

    @field_validator("provider", "model")
    @classmethod
    def _blank(cls, v: str | None) -> str | None:
        return v.strip() or None if v else None


def build_config(
    settings: RunSettings,
    run_dir: Path,
    *,
    base_config: str | Path | None = None,
    defaults: dict[str, Any] | None = None,
) -> Config:
    """Server config file < server CLI defaults (provider/model) < form settings."""
    gen: dict[str, Any] = {"question_types": settings.question_types}
    if settings.num_questions is not None:
        gen["num_questions"] = settings.num_questions
    if settings.best is not None:
        gen["select_top"] = settings.best
    if settings.seed is not None:
        gen["seed"] = settings.seed
    personas = [p.strip() for p in settings.personas if p.strip()]
    if personas:
        gen["personas"] = personas
    default_model: dict[str, Any] = dict(defaults or {})
    if settings.provider:
        default_model["provider"] = settings.provider
    if settings.model:
        default_model["model"] = settings.model
    models: dict[str, Any] = {"default": default_model}
    if default_model.get("provider") == "mock":
        models["embeddings"] = {"provider": "hashing"}
    overrides: dict[str, Any] = {
        "inputs": [str(run_dir / "sources")],
        "output": str(run_dir / "questions.json"),
        "jsonl_dir": str(run_dir / "bundle"),
        "mode": settings.mode,
        "generation": gen,
        "models": models,
    }
    if settings.max_chunks:
        overrides["extraction"] = {"max_chunks": settings.max_chunks}
    return Config.load(base_config, **overrides)


class _ThreadLogHandler(logging.Handler):
    """Copies records emitted on one thread to a file."""

    def __init__(self, path: Path, thread_id: int) -> None:
        super().__init__(logging.INFO)
        self.thread_id = thread_id
        self.fh = open(path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        try:
            self.fh.write(self.format(record) + "\n")
            self.fh.flush()
        except Exception:  # pragma: no cover
            self.handleError(record)

    def close(self) -> None:
        self.fh.close()
        super().close()


Runner = Callable[[Config], Any]


async def _generate(cfg: Config) -> dict[str, Any]:
    from trivium.orchestration.controller import generate_dataset

    dataset, run = await generate_dataset(cfg)
    return {
        "accepted": len(dataset.questions),
        "candidates": run.n_candidates,
        "rounds": run.rounds,
        "stop_reason": run.stop_reason,
        "llm_calls": run.llm_calls,
    }


class JobQueue:
    def __init__(
        self,
        workspace: Workspace,
        *,
        base_config: str | Path | None = None,
        defaults: dict[str, Any] | None = None,
        runner: Callable[[Config], Any] | None = None,
    ) -> None:
        self.ws = workspace
        self.base_config = base_config
        self.defaults = defaults or {}
        self._runner = runner or (lambda cfg: asyncio.run(_generate(cfg)))
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.current: str | None = None
        self._mark_interrupted()

    def _mark_interrupted(self) -> None:
        """Jobs left queued or running by a previous server process will never finish."""
        for d in self.ws.root.iterdir():
            job = self.ws.job(d) if d.is_dir() else None
            if job and job.get("status") in ("queued", "running"):
                self.ws.update_job(d, status="interrupted", finished_at=now())

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, name="trivium-jobs", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=1)

    def submit(self, run_id: str) -> None:
        self.ws.update_job(self.ws.run_dir(run_id), status="queued", queued_at=now(), error=None)
        self._queue.put(run_id)
        self.start()

    def position(self, run_id: str) -> int | None:
        with self._queue.mutex:
            pending = [r for r in self._queue.queue if r is not None]
        return pending.index(run_id) + 1 if run_id in pending else None

    def join(self, timeout: float | None = None) -> None:
        """Wait until every submitted job has finished (tests)."""
        done = threading.Event()

        def waiter() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=waiter, daemon=True).start()
        done.wait(timeout)

    def _work(self) -> None:
        while True:
            run_id = self._queue.get()
            try:
                if run_id is None:
                    return
                self._run(run_id)
            finally:
                self._queue.task_done()

    def _run(self, run_id: str) -> None:
        run_dir = self.ws.run_dir(run_id)
        job = self.ws.job(run_dir) or {}
        self.current = run_id
        handler = _ThreadLogHandler(run_dir / "generate.log", threading.get_ident())
        logger = logging.getLogger("trivium")
        level = logger.level
        logger.addHandler(handler)
        try:
            cfg = build_config(
                RunSettings.model_validate(job.get("settings", {})),
                run_dir,
                base_config=self.base_config,
                defaults=self.defaults,
            )
            self.ws.update_job(run_dir, status="running", started_at=now())
            logger.setLevel(logging.INFO)
            log.info("job %s: generating from %s", run_id, ", ".join(job.get("files", [])))
            summary = self._runner(cfg)
            self.ws.update_job(run_dir, status="done", finished_at=now(), summary=summary)
            log.info("job %s: done", run_id)
        except Exception as exc:
            log.error("job %s failed: %s\n%s", run_id, exc, traceback.format_exc())
            self.ws.update_job(
                run_dir, status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}"
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(level)
            handler.close()
            self.current = None
