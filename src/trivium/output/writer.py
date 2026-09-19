"""JSON / JSONL writers (design §26–27)."""

from __future__ import annotations

import json
from pathlib import Path

from trivium.evaluation.evaluator import EvaluatedCandidate
from trivium.facts.store import FactStore
from trivium.ingestion.models import Chunk, Document
from trivium.output.schema import Dataset


def write_json(dataset: Dataset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    return path


def write_jsonl_dir(
    dataset: Dataset,
    out_dir: str | Path,
    *,
    store: FactStore | None = None,
    documents: list[Document] | None = None,
    chunks: list[Chunk] | None = None,
    rejected: list[EvaluatedCandidate] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dataset.model_dump(mode="json", exclude={"questions"}, exclude_none=True)
    (out / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(out / "questions.jsonl", "w", encoding="utf-8") as fh:
        for q in dataset.questions:
            fh.write(q.model_dump_json(exclude_none=True) + "\n")
    if store is not None:
        store.save_jsonl(out / "facts.jsonl")
    if documents is not None:
        with open(out / "documents.jsonl", "w", encoding="utf-8") as fh:
            for d in documents:
                fh.write(d.model_dump_json() + "\n")
    if chunks is not None:
        with open(out / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(c.model_dump_json() + "\n")
    if rejected is not None:
        with open(out / "rejected.jsonl", "w", encoding="utf-8") as fh:
            for ec in rejected:
                fh.write(ec.model_dump_json(exclude_none=True) + "\n")
    return out
