"""User-facing dataset schema (design §26, §34)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from trivium.evaluation.evaluator import EvaluatedCandidate, Evaluation, EvidenceSpan


class GenerationInfo(BaseModel):
    strategy: str
    round: int
    source_fact_ids: list[str]
    target_difficulty: str | None = None
    graph_path: list[str] | None = None
    derivation: dict[str, Any] | None = None
    reasoning_depth: int = 1


class QuestionRecord(BaseModel):
    id: str
    question: str
    type: str
    options: list[str] | None = None
    answer: str
    accepted_answers: list[str]
    difficulty: str
    generation: GenerationInfo
    evidence: list[EvidenceSpan]
    evaluation: Evaluation
    provenance: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_evaluated(cls, ec: EvaluatedCandidate, qid: str) -> QuestionRecord:
        c = ec.candidate
        return cls(
            id=qid,
            question=c.question,
            type=c.question_type.value,
            options=c.options,
            answer=c.answer,
            accepted_answers=c.all_answers(),
            difficulty=ec.evaluation.difficulty.value,
            generation=GenerationInfo(
                strategy=c.strategy,
                round=c.round,
                source_fact_ids=c.source_fact_ids,
                target_difficulty=c.target_difficulty,
                graph_path=c.graph_path,
                derivation=c.derivation,
                reasoning_depth=c.reasoning_depth,
            ),
            evidence=ec.evidence,
            evaluation=ec.evaluation,
            provenance=ec.provenance,
        )


class DatasetInfo(BaseModel):
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    source_type: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    requested_questions: int
    generated_candidates: int
    accepted_questions: int
    rounds: int
    selected_top: int | None = None  # --best K: pool ranked, top-k kept
    selected_by: str | None = None
    stop_reason: str
    mode: str
    models: dict[str, str] = Field(default_factory=dict)
    prompt_version: str = ""
    seed: int = 0
    llm_calls: int = 0
    config: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    dataset: DatasetInfo
    dataset_metrics: dict[str, Any] = Field(default_factory=dict)
    coverage: dict[str, Any] = Field(default_factory=dict)
    strategy_stats: dict[str, Any] = Field(default_factory=dict)
    questions: list[QuestionRecord]


def dataset_metrics(
    records: list[QuestionRecord], n_candidates: int, coverage: dict[str, Any]
) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"num_questions": 0, "candidate_acceptance_rate": 0.0}

    def mean(key: str) -> float | None:
        vals = [
            getattr(r.evaluation, key) for r in records if getattr(r.evaluation, key) is not None
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

    def dist(key) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in records:
            k = key(r)
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    return {
        "num_questions": n,
        "difficulty_distribution": dist(lambda r: r.difficulty),
        "type_distribution": dist(lambda r: r.type),
        "strategy_distribution": dist(lambda r: r.generation.strategy),
        "retrieval_difficulty_distribution": dist(
            lambda r: r.evaluation.retrieval_difficulty or "unknown"
        ),
        "average_quality": mean("overall_quality"),
        "average_document_dependence": mean("document_dependence"),
        "average_closed_book_accuracy": mean("closed_book_accuracy"),
        "average_open_book_accuracy": mean("open_book_accuracy"),
        "average_novelty": mean("novelty"),
        "average_lexical_overlap": mean("lexical_overlap"),
        "fact_coverage": coverage.get("fact_coverage"),
        "section_coverage": coverage.get("section_coverage"),
        "document_coverage": coverage.get("document_coverage"),
        "multi_hop_fraction": round(
            sum(1 for r in records if r.generation.reasoning_depth >= 2) / n, 4
        ),
        "cross_document_fraction": round(
            sum(1 for r in records if len({e.document_id for e in r.evidence}) >= 2) / n, 4
        ),
        "candidate_acceptance_rate": round(n / n_candidates, 4) if n_candidates else 0.0,
    }
