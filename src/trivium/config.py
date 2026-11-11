"""Configuration for the generation pipeline (YAML file + CLI overrides)."""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field, model_validator


class Mode(str, Enum):
    rag = "rag"
    trivia = "trivia"


class QuestionType(str, Enum):
    short_answer = "short_answer"
    true_false = "true_false"
    multiple_choice = "multiple_choice"


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"
    extreme = "extreme"  # even the largest evaluator fails with gold evidence
    unknown = "unknown"


# ----------------------------------------------------------------------------- models
class ModelSpec(BaseModel):
    provider: str = "openai"  # openai | anthropic | argo | asksage | alcf | mock
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2048
    max_concurrency: int = 8
    mock_skill: float = 1.0  # only for provider=mock: ability of the simulated answerer
    # provider-specific settings passed through to the client, e.g. for argo:
    #   {user: <anl username>, env: prod|dev|test}  (user defaults to $ARGO_USER)
    options: dict[str, Any] = Field(default_factory=dict)

    def key(self) -> str:
        """Identity of the client instance: every field that changes behaviour."""
        opts = json.dumps(self.options, sort_keys=True)
        return "|".join(
            str(x)
            for x in (
                self.provider,
                self.model,
                self.base_url,
                self.api_key_env,
                self.temperature,
                self.max_tokens,
                self.max_concurrency,
                self.mock_skill,
                opts,
            )
        )

    def merged(self, override: dict[str, Any] | ModelSpec | None) -> ModelSpec:
        """Return a validated copy with ``override`` applied; ``options`` dicts are merged."""
        if override is None:
            return self
        data = (
            override.model_dump(exclude_unset=True)
            if isinstance(override, ModelSpec)
            else dict(override)
        )
        merged_opts = {**self.options, **(data.pop("options", None) or {})}
        return ModelSpec.model_validate({**self.model_dump(), **data, "options": merged_opts})


class EmbeddingSpec(BaseModel):
    provider: str = "hashing"  # hashing | openai | argo | asksage | alcf | sentence-transformers
    model: str | None = None
    base_url: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    """Per-role model assignment (design §29).

    ``default`` is used for every role that has no entry in ``roles``. Entries in
    ``roles`` may be partial (e.g. only ``model``) and are merged onto ``default``.
    ``difficulty_small`` / ``difficulty_medium`` are optional; without them difficulty is
    estimated from closed-book accuracy and reasoning depth.
    """

    default: ModelSpec = Field(default_factory=ModelSpec)
    roles: dict[str, dict[str, Any] | None] = Field(default_factory=dict)
    embeddings: EmbeddingSpec = Field(default_factory=EmbeddingSpec)

    OPTIONAL_ROLES: ClassVar[frozenset[str]] = frozenset({"difficulty_small", "difficulty_medium"})

    def spec_for(self, role: str) -> ModelSpec | None:
        if role in self.roles:
            override = self.roles[role]
            if override is None:
                return None
            return self.default.merged(override)
        if role in self.OPTIONAL_ROLES:
            return None
        return self.default


# ------------------------------------------------------------------------- pipeline
class ChunkingConfig(BaseModel):
    target_chars: int = 1500
    max_chars: int = 2500
    min_chars: int = 200


class ExtractionConfig(BaseModel):
    validate_facts: bool = True
    min_confidence: float = 0.6
    min_quality: float = 0.7
    max_facts_per_chunk: int = 15
    max_chunks: int | None = None  # debugging aid: only extract from the first N chunks


class Thresholds(BaseModel):
    """Hard acceptance constraints (design §25)."""

    evidence_entailment: float = 0.9
    answerability: float = 0.9
    ambiguity: float = 0.2
    answer_leakage: float = 0.5
    open_book_accuracy: float = 0.8
    duplicate_similarity: float = 0.90
    overall_quality: float = 0.75
    # closed-book accuracy above this rejects the question in strict RAG mode
    closed_book_max_for_rag: float = 0.6
    min_document_dependence: float = 0.3
    # difficulty calibration (design §16)
    tier_pass_accuracy: float = 0.8
    # rubric: the question must make sense to a reader who has not seen the source
    standalone: float = 0.6


class EvaluationConfig(BaseModel):
    thresholds: Thresholds = Field(default_factory=Thresholds)
    closed_book_trials: int = 3
    open_book_trials: int = 1
    difficulty_trials: int = 2
    use_llm_judge: bool = True
    retrieval_difficulty: bool = True
    retrieval_ranks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    # composite quality weights per mode (design §24)
    weights: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "rag": {
                "evidence_entailment": 0.25,
                "answerability": 0.20,
                "clarity": 0.15,
                "document_dependence": 0.15,
                "novelty": 0.10,
                "specificity": 0.10,
                "interestingness": 0.05,
            },
            "trivia": {
                "evidence_entailment": 0.20,
                "answerability": 0.20,
                "clarity": 0.20,
                "document_dependence": 0.05,
                "novelty": 0.10,
                "specificity": 0.10,
                "interestingness": 0.15,
            },
        }
    )


class GenerationConfig(BaseModel):
    num_questions: int | None = None
    max_questions: int = 500  # cap for the automatic target (design §4)
    easy: int | None = None
    medium: int | None = None
    hard: int | None = None
    question_types: list[QuestionType] = Field(default_factory=lambda: [QuestionType.short_answer])
    type_ratio: dict[str, float] | None = None  # e.g. {"short_answer": .6, "true_false": .2, ...}
    max_answer_words: int = 8
    max_statement_words: int = 30  # true/false statements longer than this are discarded
    require_document_dependence: bool = False
    multi_hop_ratio: float = 0.15
    cross_document_ratio: float = 0.05
    seed: int = 0
    batch_size: int = 8
    max_rounds: int = 60
    patience: int = 4  # rounds without an acceptance before stopping (exhaustion)
    min_novelty: float = 0.15  # stop when candidate novelty stays below this
    max_candidates: int | None = None  # optional cost budget
    # best-of selection (CLI --best K): generate an accepted pool, rank it, keep the top-k
    select_top: int | None = None
    select_by: str | None = None  # challenge | interest | trivia | quality (None → by mode)
    pool_factor: int = 8  # pool size = max(10, pool_factor * select_top) when num_questions unset
    # steer the adaptive loop toward a difficulty from round one (implied "hard" by select_top)
    prefer_difficulty: str | None = None
    # personas (CLI --persona): who asks. Steers fact scoring, question writing and the ranking
    # tournament toward what this audience needs, and enables the gap-driven `persona` strategy
    # (questions asked from the table of contents only, then grounded in the text).
    personas: list[str] = Field(default_factory=list)
    persona_weight: float = 0.5  # base weight of the `persona` strategy when personas are set
    persona_ideas_per_call: int = 8  # questions proposed per ideation call
    strategy_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "atomic": 0.25,
            "relationship": 0.25,
            "comparison": 0.20,
            "derived": 0.10,
            "multi_hop": 0.15,
            "cross_document": 0.05,
        }
    )

    @model_validator(mode="after")
    def _quota_consistency(self) -> GenerationConfig:
        quotas = [q for q in (self.easy, self.medium, self.hard) if q is not None]
        if quotas and self.num_questions is None:
            self.num_questions = sum(quotas)
        elif quotas and self.num_questions is not None and sum(quotas) > self.num_questions:
            raise ValueError(
                f"difficulty quotas sum to {sum(quotas)} but num_questions is {self.num_questions}"
            )
        if self.select_top is not None:
            if self.select_top < 1:
                raise ValueError("select_top must be >= 1")
            if self.num_questions is None:
                self.num_questions = min(
                    self.max_questions, max(10, self.pool_factor * self.select_top)
                )
            elif self.num_questions < self.select_top:
                raise ValueError("num_questions (the pool) must be >= select_top")
            if self.prefer_difficulty is None:
                self.prefer_difficulty = "hard"
        if self.prefer_difficulty is not None and self.prefer_difficulty not in (
            "easy",
            "medium",
            "hard",
        ):
            raise ValueError("prefer_difficulty must be easy, medium or hard")
        if self.select_by is not None and self.select_by not in (
            "challenge",
            "interest",
            "trivia",
            "quality",
        ):
            raise ValueError("select_by must be challenge, interest, trivia or quality")
        return self

    def difficulty_quota(self) -> dict[str, int] | None:
        if all(q is None for q in (self.easy, self.medium, self.hard)):
            return None
        return {"easy": self.easy or 0, "medium": self.medium or 0, "hard": self.hard or 0}


class TriviaConfig(BaseModel):
    """Settings that only apply in ``mode: trivia``.

    Trivia questions should be about what makes a document notable, have crisp answers, make
    sense without the source, and (for multiple choice) not be guessable from the options alone.
    """

    # score every fact for trivia value and steer sampling toward high scorers
    # (None = on in trivia mode, off otherwise)
    score_facts: bool | None = None
    brief_chars: int = 9000  # document text shown to the brief writer (head + tail)
    score_batch: int = 40  # facts per scoring call
    # defaults applied in trivia mode unless generation/extraction set them explicitly
    max_answer_words: int = 4
    max_facts_per_chunk: int = 8
    max_statement_words: int = 25
    # give the question writer the surrounding passage and the document brief
    passage_context: bool = True
    passage_chars: int = 1500
    # reject a multiple-choice form whose chance-corrected closed-book accuracy exceeds this
    # (the short-answer form is kept instead)
    mc_max_closed_book: float = 0.67
    # Swiss-style pairwise "which is the better trivia question" rounds used by --best ranking
    pairwise_rounds: int = 3


class CacheConfig(BaseModel):
    enabled: bool = True
    dir: str = ".trivium_cache"


class Config(BaseModel):
    inputs: list[str] = Field(default_factory=list)
    topic: str | None = None
    topic_max_sources: int = 5
    output: str = "questions.json"
    jsonl_dir: str | None = None
    mode: Mode = Mode.rag
    log_level: str = "INFO"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    trivia: TriviaConfig = Field(default_factory=TriviaConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @model_validator(mode="after")
    def _trivia_defaults(self) -> Config:
        """Trivia mode tightens answer length, statement length and extraction granularity
        unless the user set those values explicitly."""
        if self.mode == Mode.trivia:
            g, t = self.generation, self.trivia
            if "max_answer_words" not in g.model_fields_set:
                g.max_answer_words = t.max_answer_words
            if "max_statement_words" not in g.model_fields_set:
                g.max_statement_words = t.max_statement_words
            if "max_facts_per_chunk" not in self.extraction.model_fields_set:
                self.extraction.max_facts_per_chunk = t.max_facts_per_chunk
        return self

    @property
    def is_trivia(self) -> bool:
        return self.mode == Mode.trivia

    @property
    def score_trivia_facts(self) -> bool:
        return self.trivia.score_facts if self.trivia.score_facts is not None else self.is_trivia

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> Config:
        data: dict[str, Any] = {}
        if path is not None:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        # precedence: explicit overrides (CLI) > environment > YAML file > defaults
        data = _deep_update(data, _env_overrides())
        data = _deep_update(data, overrides)
        return cls.model_validate(data)

    def public_dump(self) -> dict[str, Any]:
        """Config as embedded in output datasets: no paths, no per-user credentials."""
        data = self.model_dump(mode="json", exclude={"inputs", "output", "jsonl_dir"})
        models = data.get("models", {})
        specs = [models.get("default", {})] + list(models.get("roles", {}).values())
        for spec in specs:
            opts = spec.get("options") if isinstance(spec, dict) else None
            if isinstance(opts, dict):
                for k in list(opts):
                    if k in _SENSITIVE_OPTIONS:
                        opts[k] = "<redacted>"
        emb_opts = models.get("embeddings", {}).get("options")
        if isinstance(emb_opts, dict):
            for k in list(emb_opts):
                if k in _SENSITIVE_OPTIONS:
                    emb_opts[k] = "<redacted>"
        return data

    @property
    def effective_select_by(self) -> str:
        """``generation.select_by`` or the mode default (trivia → hard *and* interesting)."""
        if self.generation.select_by:
            return self.generation.select_by
        return "trivia" if self.mode == Mode.trivia else "challenge"

    @property
    def cache_path(self) -> Path:
        return Path(self.cache.dir) / "responses.sqlite"

    @property
    def quality_weights(self) -> dict[str, float]:
        return self.evaluation.weights[self.mode.value]

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


_SENSITIVE_OPTIONS = {"user", "api_key", "token", "password", "secret"}


def _env_overrides() -> dict[str, Any]:
    """``TRIVIUM_PROVIDER`` / ``TRIVIUM_MODEL`` / ``TRIVIUM_BASE_URL`` for the default model."""
    default: dict[str, Any] = {}
    for env, field in (
        ("TRIVIUM_PROVIDER", "provider"),
        ("TRIVIUM_MODEL", "model"),
        ("TRIVIUM_BASE_URL", "base_url"),
    ):
        if os.environ.get(env):
            default[field] = os.environ[env]
    return {"models": {"default": default}} if default else {}


def _deep_update(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out
