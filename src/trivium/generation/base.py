"""Shared candidate representation, generation context and generator base class."""

from __future__ import annotations

import logging
import random
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from trivium.config import Config, QuestionType
from trivium.facts.models import Fact
from trivium.facts.store import FactStore, norm
from trivium.graph.backend import GraphBackend
from trivium.ingestion.models import Chunk
from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ data models
class Candidate(BaseModel):
    id: str
    question: str
    answer: str
    question_type: QuestionType = QuestionType.short_answer
    answer_type: str = "short_answer"  # short_answer | numeric | boolean | choice
    accepted_answers: list[str] = Field(default_factory=list)
    options: list[str] | None = None  # multiple choice
    source_fact_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)
    strategy: str
    round: int = 0
    target_difficulty: str | None = None
    graph_path: list[str] | None = None
    reasoning_depth: int = 1
    derivation: dict[str, Any] | None = None
    rationale: str = ""

    def all_answers(self) -> list[str]:
        seen: list[str] = []
        for a in [self.answer, *self.accepted_answers]:
            if a and a not in seen:
                seen.append(a)
        return seen


class Feedback(BaseModel):
    """What the coverage/rejection analysis tells the next generation round (design §21)."""

    target_difficulty: str | None = None
    priority_sections: list[list[str]] = Field(default_factory=list)  # [document_id, section_id]
    priority_documents: list[str] = Field(default_factory=list)
    overused_fact_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    rejection_summary: dict[str, int] = Field(default_factory=dict)

    def as_prompt_text(self) -> str:
        if not self.notes and not self.target_difficulty:
            return ""
        lines = ["Guidance from previous rounds:"]
        if self.target_difficulty:
            lines.append(f"- Aim for {self.target_difficulty} difficulty.")
        lines.extend(f"- {n}" for n in self.notes)
        return "\n".join(lines)


class GeneratedQuestion(BaseModel):
    question: str
    answer: str
    accepted_answers: list[str] = Field(default_factory=list)
    answer_type: str = "short_answer"
    rationale: str = ""


# --------------------------------------------------------------------- context
@dataclass
class GenerationContext:
    config: Config
    store: FactStore
    graph: GraphBackend
    chunks: dict[str, Chunk]
    llm: LLM
    rng: random.Random
    accepted: list[Candidate] = field(default_factory=list)
    used_facts: Counter = field(default_factory=Counter)
    feedback: Feedback = field(default_factory=Feedback)
    round: int = 0
    _counter: int = 0

    def next_id(self) -> str:
        self._counter += 1
        return f"cand_{self._counter:05d}"

    @property
    def max_answer_words(self) -> int:
        return self.config.generation.max_answer_words

    # ---------------------------------------------------------- fact sampling
    def fact_weight(self, fact: Fact) -> float:
        if fact.id in self.feedback.overused_fact_ids:
            return 0.05  # coverage feedback: already used by several accepted questions
        w = 0.25 + fact.quality_score
        uses = self.used_facts.get(fact.id, 0)
        w /= 1.0 + 2.0 * uses
        if [fact.document_id, fact.section_id] in self.feedback.priority_sections:
            w *= 3.0
        elif fact.document_id in self.feedback.priority_documents:
            w *= 2.0
        if fact.object.is_numeric or fact.fact_type.value in (
            "result",
            "quantity",
            "event",
            "temporal",
        ):
            w *= 1.3
        return w

    def sample_facts(
        self, n: int, pool: Iterable[Fact] | None = None, *, distinct_chunks: bool = True
    ) -> list[Fact]:
        facts = list(pool) if pool is not None else self.store.all()
        if not facts:
            return []
        weights = [self.fact_weight(f) for f in facts]
        chosen: list[Fact] = []
        seen_chunks: set[str] = set()
        tries = 0
        while len(chosen) < n and tries < 20 * n and facts:
            tries += 1
            f = self.rng.choices(facts, weights=weights, k=1)[0]
            if f in chosen or (distinct_chunks and f.chunk_id in seen_chunks and len(facts) > n):
                continue
            chosen.append(f)
            seen_chunks.add(f.chunk_id)
        return chosen

    def evidence_for(self, facts: list[Fact]) -> list[dict[str, str]]:
        return [
            {
                "fact_id": f.id,
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object.display(),
                "qualifiers": f.qualifiers,
                "statement": f.statement(),
                "evidence": f.evidence,
                "document_id": f.document_id,
                "section": self.chunks[f.chunk_id].section_title
                if f.chunk_id in self.chunks
                else f.section_id,
            }
            for f in facts
        ]

    def make_candidate(
        self, gq: GeneratedQuestion, facts: list[Fact], strategy: str, **extra: Any
    ) -> Candidate:
        return Candidate(
            id=self.next_id(),
            question=gq.question.strip(),
            answer=gq.answer.strip(),
            accepted_answers=[a.strip() for a in gq.accepted_answers if a.strip()],
            answer_type=gq.answer_type
            if gq.answer_type in ("short_answer", "numeric")
            else "short_answer",
            source_fact_ids=[f.id for f in facts],
            source_chunk_ids=sorted({f.chunk_id for f in facts}),
            source_document_ids=sorted({f.document_id for f in facts}),
            strategy=strategy,
            round=self.round,
            target_difficulty=self.feedback.target_difficulty,
            rationale=gq.rationale,
            **extra,
        )


# ------------------------------------------------------------------- generator
QUESTION_SYSTEM = (
    "You write precise, self-contained questions whose answers are short and objectively gradable."
)

QUESTION_RULES = """Rules for the question:
- It must be answerable *only* from the given facts/evidence and must have exactly one correct answer.
- Include enough context (entity names, dataset, conditions) to make the answer unique, but do not
  restate the answer or use words that give it away.
- The answer must be at most {max_words} words: a name, number with unit, date, or short phrase.
- Do not refer to "the text", "the paper", "the document" or "the author"; ask about the content itself.
- Prefer questions a knowledgeable reader could not answer without these specific sources.
{extra}"""


class Generator:
    """Base class: subclasses implement :meth:`propose`, returning candidates for one batch."""

    name: str = "base"
    question_type: QuestionType = QuestionType.short_answer
    reasoning_depth: int = 1

    def __init__(self, ctx: GenerationContext):
        self.ctx = ctx

    async def generate(self, n: int) -> list[Candidate]:
        try:
            cands = await self.propose(n)
        except Exception as exc:  # noqa: BLE001 — a failing strategy must not kill the loop
            log.exception("generator %s failed: %s", self.name, exc)
            return []
        for c in cands:
            c.strategy = self.name
            c.question_type = c.question_type or self.question_type
        return cands

    async def propose(self, n: int) -> list[Candidate]:
        raise NotImplementedError

    def available(self) -> bool:
        """Whether this strategy can produce anything with the current knowledge base."""
        return len(self.ctx.store) > 0

    # ------------------------------------------------------------- LLM helper
    async def ask_llm(
        self,
        facts: list[Fact],
        instructions: str,
        *,
        fixed_answer: str | None = None,
        extra_rules: str = "",
        payload_extra: dict[str, Any] | None = None,
        task: str = "generate_question",
    ) -> GeneratedQuestion | None:
        ctx = self.ctx
        payload: dict[str, Any] = {
            "strategy": self.name,
            "facts": ctx.evidence_for(facts),
            "max_answer_words": ctx.max_answer_words,
        }
        if fixed_answer is not None:
            payload["required_answer"] = fixed_answer
        if payload_extra:
            payload.update(payload_extra)
        rules = QUESTION_RULES.format(max_words=ctx.max_answer_words, extra=extra_rules)
        guidance = ctx.feedback.as_prompt_text()
        text = "\n\n".join(p for p in (instructions, rules, guidance) if p)
        try:
            resp = await ctx.llm.generate(
                build_messages(QUESTION_SYSTEM, text, payload), GeneratedQuestion, task=task
            )
        except LLMError as exc:
            log.warning("%s: question generation failed: %s", self.name, exc)
            return None
        gq: GeneratedQuestion = resp.parsed  # type: ignore[assignment]
        if fixed_answer is not None:
            # the phrasing model must never change a deterministically derived answer; its own
            # phrasing is kept as an accepted alias only when it is equivalent
            if answers_match(gq.answer, fixed_answer):
                gq.accepted_answers = [a for a in [gq.answer, *gq.accepted_answers] if a]
            else:
                log.debug(
                    "LLM answer %r differs from fixed answer %r; overriding",
                    gq.answer,
                    fixed_answer,
                )
                gq.accepted_answers = []
            gq.answer = fixed_answer
        if len(gq.answer.split()) > ctx.max_answer_words or not gq.question.strip():
            return None
        return gq


_WORD = re.compile(r"[a-z0-9.]+")


def answers_match(a: str, b: str) -> bool:
    """Token-based equivalence of two short answers.

    Equal after normalisation, or one token sequence is a contiguous *token* sub-sequence of the
    other ("0.12" vs "0.12 ev"). Plain substring tests are not used: "2.5" ≠ "12.5", "no" ≠ "know".
    """
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    ta, tb = _WORD.findall(na), _WORD.findall(nb)
    if not ta or not tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(long_) > len(short) + 3:  # a qualifier or two at most
        return False
    return any(long_[i : i + len(short)] == short for i in range(len(long_) - len(short) + 1))
