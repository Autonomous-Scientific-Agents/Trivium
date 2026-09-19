"""Full candidate evaluation pipeline with early rejection (design §12–§18, §24–§25)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from trivium.config import Config, Difficulty
from trivium.evaluation.answering import accuracy
from trivium.evaluation.difficulty import calibrate
from trivium.evaluation.quality import heuristic_leakage, score_quality
from trivium.evaluation.redundancy import RedundancyIndex
from trivium.evaluation.retrieval import RetrievalProbe
from trivium.facts.store import FactStore
from trivium.generation.base import Candidate
from trivium.ingestion.models import Chunk
from trivium.models.registry import ModelRegistry
from trivium.prompts import PROMPT_VERSION

log = logging.getLogger(__name__)


class Evaluation(BaseModel):
    answerability: float | None = None
    evidence_entailment: float | None = None
    clarity: float | None = None
    specificity: float | None = None
    ambiguity: float | None = None
    answer_conciseness: float | None = None
    answer_leakage: float | None = None
    interestingness: float | None = None

    closed_book_accuracy: float | None = None
    closed_book_chance_corrected: float | None = None
    open_book_accuracy: float | None = None
    document_dependence: float | None = None
    closed_book_answers: list[str] = Field(default_factory=list)
    open_book_answers: list[str] = Field(default_factory=list)

    tier_accuracy: dict[str, float] = Field(default_factory=dict)
    difficulty: Difficulty = Difficulty.unknown
    difficulty_method: str = ""
    retrieval_difficulty: str | None = None
    gold_rank: int | None = None
    reasoning_depth: int = 1

    max_similarity_to_existing_question: float = 0.0
    fact_overlap_with_existing: float = 0.0
    novelty: float = 1.0
    nearest_question_id: str | None = None

    overall_quality: float | None = None
    accepted: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    notes: str = ""


class EvidenceSpan(BaseModel):
    document_id: str
    section: str
    chunk_id: str
    text: str
    fact_ids: list[str] = Field(default_factory=list)


class EvaluatedCandidate(BaseModel):
    candidate: Candidate
    evaluation: Evaluation
    evidence: list[EvidenceSpan]
    provenance: dict[str, Any] = Field(default_factory=dict)


class Evaluator:
    def __init__(
        self, config: Config, models: ModelRegistry, store: FactStore, chunks: list[Chunk]
    ):
        self.cfg = config
        self.t = config.evaluation.thresholds
        self.models = models
        self.store = store
        self.chunks = {c.id: c for c in chunks}
        self.redundancy = RedundancyIndex(models.embedder)
        self.retrieval = (
            RetrievalProbe(models.embedder, chunks, config.evaluation.retrieval_ranks)
            if config.evaluation.retrieval_difficulty
            else None
        )
        self._judge = models.get("judge") if config.evaluation.use_llm_judge else None

    # -------------------------------------------------------------- evidence
    def evidence_spans(self, cand: Candidate) -> list[EvidenceSpan]:
        facts = self.store.many(cand.source_fact_ids)
        spans: dict[str, EvidenceSpan] = {}
        for f in facts:
            ch = self.chunks.get(f.chunk_id)
            key = f.chunk_id
            if key not in spans:
                spans[key] = EvidenceSpan(
                    document_id=f.document_id,
                    section=ch.section_title if ch else f.section_id,
                    chunk_id=f.chunk_id,
                    text=f.evidence,
                    fact_ids=[f.id],
                )
            else:
                sp = spans[key]
                if f.evidence not in sp.text:
                    sp.text = f"{sp.text} … {f.evidence}"
                sp.fact_ids.append(f.id)
        return list(spans.values())

    def evidence_texts(self, cand: Candidate) -> list[str]:
        """Evidence as passed to answerers: the sentence-level quotes plus their chunk text."""
        out: list[str] = []
        for cid in cand.source_chunk_ids:
            ch = self.chunks.get(cid)
            if ch:
                out.append(f"[{ch.document_id} / {ch.section_title}]\n{ch.text}")
        if not out:
            out = [sp.text for sp in self.evidence_spans(cand)]
        return out

    # --------------------------------------------------------------- scoring
    def composite(self, ev: Evaluation) -> float:
        w = self.cfg.quality_weights
        comps = {
            "evidence_entailment": ev.evidence_entailment,
            "answerability": ev.answerability,
            "clarity": ev.clarity,
            "document_dependence": None
            if ev.document_dependence is None
            else max(0.0, ev.document_dependence),
            "novelty": ev.novelty,
            "specificity": ev.specificity,
            "interestingness": ev.interestingness,
        }
        total_w = 0.0
        score = 0.0
        for k, weight in w.items():
            v = comps.get(k)
            if v is None:
                continue
            score += weight * v
            total_w += weight
        return round(score / total_w, 4) if total_w else 0.0

    async def evaluate(self, cand: Candidate) -> EvaluatedCandidate:
        ev = Evaluation(reasoning_depth=cand.reasoning_depth)
        reasons = ev.rejection_reasons
        evidence = self.evidence_texts(cand)

        # 1. structural checks (free)
        if (
            len(cand.answer.split()) > self.cfg.generation.max_answer_words
            and cand.answer_type != "choice"
        ):
            reasons.append("answer_too_long")
        # comparison questions legitimately name the compared entities, so only the LLM rubric
        # judges leakage for them; other strategies also get the cheap structural check
        if (
            cand.answer_type not in ("boolean", "choice")
            and cand.strategy != "comparison"
            and heuristic_leakage(cand) >= 1.0
        ):
            reasons.append("answer_leaked_in_question")
        sim, overlap, nearest = await self.redundancy.assess(cand, judge=self._judge)
        ev.max_similarity_to_existing_question = round(sim, 4)
        ev.fact_overlap_with_existing = round(overlap, 4)
        ev.nearest_question_id = nearest.id if nearest else None
        ev.novelty = round(max(0.0, 1.0 - max(sim, 0.6 * overlap)), 4)
        if sim >= self.t.duplicate_similarity or (
            overlap >= 1.0
            and nearest is not None
            and nearest.strategy == cand.strategy
            and sim >= 0.6
        ):
            reasons.append("duplicate")
        if reasons:
            return self._finish(cand, ev)

        # 2. quality rubric
        q = await score_quality(self.models["quality_evaluator"], cand, evidence)
        if q is None:
            reasons.append("evaluator_error")
            return self._finish(cand, ev)
        ev.answerability, ev.evidence_entailment, ev.ambiguity = (
            q.answerability,
            q.evidence_entailment,
            q.ambiguity,
        )
        ev.answer_conciseness, ev.clarity, ev.answer_leakage = (
            q.answer_conciseness,
            q.clarity,
            q.answer_leakage,
        )
        ev.specificity, ev.interestingness, ev.notes = q.specificity, q.interestingness, q.notes
        if q.evidence_entailment < self.t.evidence_entailment:
            reasons.append("unsupported_answer")
        if q.answerability < self.t.answerability:
            reasons.append("unanswerable")
        if q.ambiguity > self.t.ambiguity:
            reasons.append("ambiguous")
        if q.answer_leakage > self.t.answer_leakage:
            reasons.append("answer_leakage")
        if reasons:
            return self._finish(cand, ev)

        # 3. open-book (strong model + gold evidence) and closed-book, concurrently
        ecfg = self.cfg.evaluation
        open_task = accuracy(
            self.models["open_book"],
            cand,
            evidence=evidence,
            trials=ecfg.open_book_trials,
            judge=self._judge,
        )
        closed_task = accuracy(
            self.models["closed_book"],
            cand,
            evidence=None,
            trials=ecfg.closed_book_trials,
            judge=self._judge,
        )
        (
            (ev.open_book_accuracy, ev.open_book_answers),
            (ev.closed_book_accuracy, ev.closed_book_answers),
        ) = await asyncio.gather(open_task, closed_task)
        # chance-correct closed-book accuracy for boolean / choice items so that a coin flip does
        # not register as prior knowledge (design §15, refinement)
        chance = 0.0
        if cand.answer_type == "boolean":
            chance = 0.5
        elif cand.answer_type == "choice" and cand.options:
            chance = 1.0 / len(cand.options)
        closed_adj = (
            max(0.0, (ev.closed_book_accuracy - chance) / (1.0 - chance))
            if chance
            else ev.closed_book_accuracy
        )
        ev.closed_book_chance_corrected = round(closed_adj, 4)
        ev.document_dependence = round(ev.open_book_accuracy - closed_adj, 4)
        if ev.open_book_accuracy < self.t.open_book_accuracy:
            reasons.append("large_model_failed_open_book")
        if self.cfg.generation.require_document_dependence:
            if closed_adj > self.t.closed_book_max_for_rag:
                reasons.append("closed_book_too_easy")
            elif ev.document_dependence < self.t.min_document_dependence:
                reasons.append("low_document_dependence")
        if reasons:
            return self._finish(cand, ev)

        # 4. difficulty tiers (open-book with smaller models)
        tiers = self.models.difficulty_tiers()
        tier_results = await asyncio.gather(
            *(
                accuracy(
                    llm, cand, evidence=evidence, trials=ecfg.difficulty_trials, judge=self._judge
                )
                for llm in tiers.values()
            )
        )
        ev.tier_accuracy = {
            name: round(acc, 4) for name, (acc, _) in zip(tiers, tier_results, strict=True)
        }
        ev.tier_accuracy.setdefault("large", round(ev.open_book_accuracy, 4))
        ev.difficulty, ev.difficulty_method = calibrate(
            ev.tier_accuracy,
            threshold=self.t.tier_pass_accuracy,
            closed_book_accuracy=ev.closed_book_accuracy,
            reasoning_depth=cand.reasoning_depth,
        )
        if ev.difficulty == Difficulty.extreme:
            reasons.append("large_model_failed_open_book")

        # 5. retrieval difficulty
        if self.retrieval is not None:
            ev.gold_rank = await self.retrieval.gold_rank(cand.question, cand.source_chunk_ids)
            ev.retrieval_difficulty = self.retrieval.label(ev.gold_rank)

        # 6. composite
        ev.overall_quality = self.composite(ev)
        if ev.overall_quality < self.t.overall_quality:
            reasons.append("low_quality_score")
        return self._finish(cand, ev)

    def _finish(self, cand: Candidate, ev: Evaluation) -> EvaluatedCandidate:
        ev.accepted = not ev.rejection_reasons
        if ev.overall_quality is None and ev.evidence_entailment is not None:
            ev.overall_quality = self.composite(ev)
        prov = {
            "generation_model": f"{self.models['question_generator'].provider}:{self.models['question_generator'].model}",
            "evaluation_models": self.models.describe(),
            "prompt_version": PROMPT_VERSION,
            "seed": self.cfg.generation.seed,
            "mode": self.cfg.mode.value,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        return EvaluatedCandidate(
            candidate=cand, evaluation=ev, evidence=self.evidence_spans(cand), provenance=prov
        )

    async def register_accepted(self, cand: Candidate) -> None:
        await self.redundancy.add(cand)
