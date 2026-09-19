"""The adaptive generation loop (design §20) and the end-to-end pipeline."""

from __future__ import annotations

import asyncio
import logging
import random
from collections import Counter
from dataclasses import dataclass, field

from trivium.config import Config, Difficulty, QuestionType
from trivium.evaluation.evaluator import EvaluatedCandidate, Evaluator
from trivium.facts.extractor import extract_facts
from trivium.facts.store import FactStore
from trivium.facts.validator import validate_facts
from trivium.generation import GENERATORS, GenerationContext, Generator, MultipleChoiceConverter
from trivium.graph.builder import build_graph
from trivium.ingestion.chunker import chunk_documents
from trivium.ingestion.loader import load_inputs
from trivium.ingestion.models import Chunk, Document
from trivium.models.registry import ModelRegistry
from trivium.orchestration import coverage as cov
from trivium.orchestration.stopping import LoopState, estimate_target, should_stop
from trivium.orchestration.strategy import StrategySelector
from trivium.output.schema import Dataset, DatasetInfo, QuestionRecord, dataset_metrics
from trivium.output.writer import write_json, write_jsonl_dir
from trivium.prompts import PROMPT_VERSION

log = logging.getLogger(__name__)


@dataclass
class GenerationRun:
    config: Config
    documents: list[Document]
    chunks: list[Chunk]
    store: FactStore
    accepted: list[EvaluatedCandidate] = field(default_factory=list)
    rejected: list[EvaluatedCandidate] = field(default_factory=list)
    target: int = 0
    rounds: int = 0
    stop_reason: str = ""
    n_candidates: int = 0
    strategy_stats: dict = field(default_factory=dict)
    llm_calls: int = 0

    def to_dataset(self) -> Dataset:
        records = [
            QuestionRecord.from_evaluated(ec, f"q_{i + 1:04d}")
            for i, ec in enumerate(self.accepted)
        ]
        coverage = cov.analyze(self.store, self.accepted).model_dump(
            exclude={"fact_use_counts", "questions_per_section", "facts_per_section"}
        )
        info = DatasetInfo(
            source_type="topic" if self.config.topic else "documents",
            sources=[
                {
                    "document_id": d.id,
                    "title": d.title,
                    "source": d.source,
                    "retrieved_at": d.retrieved_at.isoformat(timespec="seconds"),
                }
                for d in self.documents
            ],
            requested_questions=self.target,
            generated_candidates=self.n_candidates,
            accepted_questions=len(records),
            rounds=self.rounds,
            stop_reason=self.stop_reason,
            mode=self.config.mode.value,
            models={},
            prompt_version=PROMPT_VERSION,
            seed=self.config.generation.seed,
            llm_calls=self.llm_calls,
            config=self.config.public_dump(),
        )
        return Dataset(
            dataset=info,
            dataset_metrics=dataset_metrics(records, self.n_candidates, coverage),
            coverage=coverage,
            strategy_stats=self.strategy_stats,
            questions=records,
        )


# --------------------------------------------------------------------- pipeline
async def build_knowledge_base(
    config: Config, models: ModelRegistry, documents: list[Document]
) -> tuple[list[Chunk], FactStore]:
    chunks = chunk_documents(documents, config.chunking)
    log.info("%d documents → %d chunks", len(documents), len(chunks))
    facts = await extract_facts(models["fact_extractor"], chunks, config.extraction)
    chunk_text = {c.id: c.text for c in chunks}
    facts = await validate_facts(models["fact_validator"], facts, chunk_text, config.extraction)
    store = FactStore(facts)
    return chunks, store


async def load_documents(config: Config) -> list[Document]:
    if config.topic:
        from trivium.topic.acquire import acquire_topic_sources

        return await acquire_topic_sources(config.topic, max_sources=config.topic_max_sources)
    if not config.inputs:
        raise ValueError("either inputs or topic must be provided")
    return load_inputs(config.inputs)


class Controller:
    def __init__(
        self,
        config: Config,
        models: ModelRegistry,
        documents: list[Document],
        chunks: list[Chunk],
        store: FactStore,
    ):
        self.cfg = config
        self.models = models
        self.run = GenerationRun(config=config, documents=documents, chunks=chunks, store=store)
        self.graph = build_graph(store)
        self.rng = random.Random(config.generation.seed)
        self.evaluator = Evaluator(config, models, store, chunks)
        self.ctx = GenerationContext(
            config=config,
            store=store,
            graph=self.graph,
            chunks={c.id: c for c in chunks},
            llm=models["question_generator"],
            rng=self.rng,
        )
        self.generators: dict[str, Generator] = {
            name: cls(self.ctx) for name, cls in GENERATORS.items()
        }
        if QuestionType.true_false not in config.generation.question_types:
            self.generators.pop("true_false", None)
        self.selector = StrategySelector(
            config.generation, {n: g.available() for n, g in self.generators.items()}
        )
        self.mc = (
            MultipleChoiceConverter(self.ctx)
            if QuestionType.multiple_choice in config.generation.question_types
            else None
        )
        self.rejections: Counter = Counter()
        self.run.target = self._target()

    def _target(self) -> int:
        g = self.cfg.generation
        if g.num_questions:
            return g.num_questions
        n_rel = sum(
            1
            for f in self.run.store
            if not f.object.is_numeric and self.graph.degree(f.object.text.lower()) > 1
        )
        n_paths = len(self.graph.two_hop_paths(max_paths=5000))
        target = estimate_target(len(self.run.store), n_rel, n_paths, g.max_questions)
        log.info(
            "auto target: %d questions (facts=%d, relational=%d, 2-hop paths=%d)",
            target,
            len(self.run.store),
            n_rel,
            n_paths,
        )
        return target

    # ------------------------------------------------------------------ loop
    async def loop(self) -> GenerationRun:
        g = self.cfg.generation
        state = LoopState()
        quota = g.difficulty_quota()
        while True:
            reason = should_stop(state, g, self.run.target)
            if reason:
                self.run.stop_reason = reason
                break
            state.round += 1
            self.ctx.round = state.round
            report = cov.analyze(self.run.store, self.run.accepted)
            self.ctx.feedback = cov.build_feedback(report, g, self.rejections, self.run.target)
            weights = self.selector.weights(
                report, self.ctx.feedback.target_difficulty, self.run.target
            )
            if not weights:
                state.strategies_available = False
                continue
            remaining = self.run.target - len(self.run.accepted)
            batch = max(1, min(g.batch_size, int(remaining * 1.5) + 1))
            alloc = self.selector.allocate(batch, weights, self.rng)
            log.info(
                "round %d: accepted %d/%d, target_difficulty=%s, batch=%s",
                state.round,
                len(self.run.accepted),
                self.run.target,
                self.ctx.feedback.target_difficulty,
                alloc,
            )

            gen_results = await asyncio.gather(
                *(self.generators[s].generate(n) for s, n in alloc.items())
            )
            candidates = []
            for (s, _), cands in zip(alloc.items(), gen_results, strict=True):
                self.selector.record_generated(s, len(cands))
                candidates.extend(cands)
            state.candidates += len(candidates)
            self.run.n_candidates += len(candidates)
            if not candidates:
                state.rounds_without_acceptance += 1
                continue

            # Evaluate candidates concurrently, but accept sequentially so that redundancy
            # checks see earlier acceptances from the same round.
            evaluated = await asyncio.gather(*(self.evaluator.evaluate(c) for c in candidates))
            n_new = 0
            novelties = []
            for ec in evaluated:
                novelties.append(ec.evaluation.novelty)
                if ec.evaluation.accepted and len(self.run.accepted) < self.run.target:
                    sim, overlap, nearest = await self.evaluator.redundancy.assess(
                        ec.candidate, judge=self.evaluator._judge
                    )
                    if self.evaluator.is_duplicate(ec.candidate, sim, overlap, nearest):
                        ec.evaluation.accepted = False
                        ec.evaluation.rejection_reasons.append("duplicate")
                    elif (
                        quota
                        and quota.get(ec.evaluation.difficulty.value, 0)
                        <= state.difficulty_counts.get(ec.evaluation.difficulty.value, 0)
                        and ec.evaluation.difficulty != Difficulty.unknown
                        and sum(quota.values()) >= self.run.target
                    ):
                        ec.evaluation.accepted = False
                        ec.evaluation.rejection_reasons.append("difficulty_quota_full")
                if ec.evaluation.accepted and len(self.run.accepted) < self.run.target:
                    ec = await self._maybe_multiple_choice(ec)
                    await self.evaluator.register_accepted(ec.candidate)
                    for fid in ec.candidate.source_fact_ids:
                        self.ctx.used_facts[fid] += 1
                    self.ctx.accepted.append(ec.candidate)
                    self.run.accepted.append(ec)
                    d = ec.evaluation.difficulty.value
                    state.difficulty_counts[d] = state.difficulty_counts.get(d, 0) + 1
                    n_new += 1
                else:
                    if ec.evaluation.accepted:  # passed every check but the target was already met
                        ec.evaluation.accepted = False
                        ec.evaluation.rejection_reasons.append("target_reached")
                    self.run.rejected.append(ec)
                    self.rejections.update(ec.evaluation.rejection_reasons)
                self.selector.record_result(
                    ec.candidate.strategy,
                    ec.evaluation.accepted,
                    ec.evaluation.rejection_reasons,
                    ec.evaluation.difficulty == Difficulty.hard,
                )
                for fid in ec.candidate.source_fact_ids:
                    if not ec.evaluation.accepted:
                        self.ctx.used_facts[fid] += (
                            0.34  # discourage re-sampling facts that keep failing
                        )
            state.accepted = len(self.run.accepted)
            state.rounds_without_acceptance = 0 if n_new else state.rounds_without_acceptance + 1
            mean_nov = sum(novelties) / len(novelties)
            state.recent_novelty.append(mean_nov)
            state.low_novelty_rounds = (
                state.low_novelty_rounds + 1 if mean_nov < g.min_novelty else 0
            )
            log.info(
                "round %d: %d candidates, %d accepted (total %d); mean novelty %.2f; rejections %s",
                state.round,
                len(candidates),
                n_new,
                len(self.run.accepted),
                mean_nov,
                dict(Counter(r for ec in evaluated for r in ec.evaluation.rejection_reasons)),
            )
        self.run.rounds = state.round
        self.run.strategy_stats = self.selector.summary()
        self.run.llm_calls = self.models.total_calls()
        log.info(
            "stopped: %s after %d rounds; %d accepted from %d candidates",
            self.run.stop_reason,
            state.round,
            len(self.run.accepted),
            self.run.n_candidates,
        )
        return self.run

    async def _maybe_multiple_choice(self, ec: EvaluatedCandidate) -> EvaluatedCandidate:
        if self.mc is None or ec.candidate.question_type != QuestionType.short_answer:
            return ec
        want = self.selector.type_target("multiple_choice")
        have = sum(
            1
            for a in self.run.accepted
            if a.candidate.question_type == QuestionType.multiple_choice
        )
        if want <= 0 or have >= want * max(1, self.run.target):
            return ec
        mc = await self.mc.convert(ec.candidate, self.evaluator.evidence_texts(ec.candidate))
        if mc is None:
            return ec
        ec_mc = await self.evaluator.evaluate(mc)
        if ec_mc.evaluation.accepted:
            return ec_mc
        self.rejections["bad_distractors"] += 1
        return ec


async def generate_dataset(config: Config, *, write: bool = True) -> tuple[Dataset, GenerationRun]:
    logging.getLogger("trivium").setLevel(config.log_level.upper())
    models = ModelRegistry(config)
    try:
        documents = await load_documents(config)
        chunks, store = await build_knowledge_base(config, models, documents)
        if len(store) == 0:
            raise RuntimeError("no validated facts were extracted; cannot generate questions")
        controller = Controller(config, models, documents, chunks, store)
        run = await controller.loop()
        dataset = run.to_dataset()
        dataset.dataset.models = models.describe()
        if write:
            write_json(dataset, config.output)
            log.info("wrote %s (%d questions)", config.output, len(dataset.questions))
            if config.jsonl_dir:
                write_jsonl_dir(
                    dataset,
                    config.jsonl_dir,
                    store=store,
                    documents=documents,
                    chunks=chunks,
                    rejected=run.rejected,
                )
                log.info("wrote JSONL bundle to %s", config.jsonl_dir)
        return dataset, run
    finally:
        await models.aclose()
        models.close()
