"""LLM-based structured fact extraction from chunks (design §7)."""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from trivium.config import ExtractionConfig
from trivium.facts.models import Fact, FactObject, FactType, looks_unresolved
from trivium.ingestion.models import Chunk
from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)


class ExtractedFact(BaseModel):
    subject: str
    predicate: str
    object: str
    value: float | None = None
    unit: str | None = None
    qualifiers: dict[str, str] = Field(default_factory=dict)
    fact_type: str = "attribute"
    explicit: bool = True
    evidence: str
    confidence: float = 0.8


class ExtractionResponse(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


EXTRACT_SYSTEM = (
    "You extract atomic, verifiable facts from text as subject / predicate / object triples."
)

EXTRACT_INSTRUCTIONS = """Extract up to {max_facts} atomic facts from the chunk below.

Rules:
- One claim per fact. Split compound sentences.
- `subject` must be a fully resolved named entity or concept (never a pronoun such as "it" or "they").
- `predicate` is a short snake_case relation (e.g. "mean_absolute_error", "trained_on", "founded_in").
- `object` is the value as text. For numeric quantities also fill `value` (number) and `unit`.
  Leave `unit` null for years, dates, counts and other unitless numbers (never use "year" as a unit).
- Put conditions that scope the claim into `qualifiers` (e.g. dataset, year, temperature).
- `fact_type` is one of: {types}.
- `evidence` is a verbatim quote (≤ 300 chars) from the chunk that supports the fact.
- `explicit` is false only if the fact requires inference beyond the text.
- `confidence` in [0, 1] reflects how certain you are that the fact is stated by the evidence.
- Prefer specific, gradable facts (numbers, names, dates, methods) over vague statements."""


def _make_fact(ex: ExtractedFact, chunk: Chunk, idx: int) -> Fact:
    try:
        ftype = FactType(ex.fact_type)
    except ValueError:
        ftype = FactType.quantity if ex.value is not None else FactType.attribute
    if ex.value is not None and ftype == FactType.attribute:
        ftype = FactType.quantity
    return Fact(
        id=f"fact_{chunk.id}_{idx:02d}",
        subject=ex.subject,
        predicate=ex.predicate.strip().lower().replace(" ", "_"),
        object=FactObject(text=ex.object.strip(), value=ex.value, unit=(ex.unit or None)),
        qualifiers={k: str(v) for k, v in ex.qualifiers.items()},
        fact_type=ftype,
        explicit=ex.explicit,
        document_id=chunk.document_id,
        section_id=chunk.section_id,
        chunk_id=chunk.id,
        evidence=ex.evidence.strip()[:400],
        confidence=max(0.0, min(1.0, ex.confidence)),
    )


async def extract_chunk(llm: LLM, chunk: Chunk, cfg: ExtractionConfig) -> list[Fact]:
    messages = build_messages(
        EXTRACT_SYSTEM,
        EXTRACT_INSTRUCTIONS.format(
            max_facts=cfg.max_facts_per_chunk,
            types=", ".join(t.value for t in FactType if t != FactType.derived),
        ),
        {
            "document_id": chunk.document_id,
            "section": chunk.section_title,
            "chunk_id": chunk.id,
            "text": chunk.text,
        },
    )
    try:
        resp = await llm.generate(messages, ExtractionResponse, task="extract_facts")
    except LLMError as exc:
        log.warning("fact extraction failed for %s: %s", chunk.id, exc)
        return []
    out: list[Fact] = []
    for i, ex in enumerate(resp.parsed.facts[: cfg.max_facts_per_chunk]):  # type: ignore[union-attr]
        if not ex.subject or not ex.object or not ex.evidence:
            continue
        if looks_unresolved(ex.subject):
            continue
        if ex.confidence < cfg.min_confidence:
            continue
        out.append(_make_fact(ex, chunk, i))
    return out


async def extract_facts(
    llm: LLM, chunks: list[Chunk], cfg: ExtractionConfig | None = None
) -> list[Fact]:
    cfg = cfg or ExtractionConfig()
    if cfg.max_chunks:
        chunks = chunks[: cfg.max_chunks]
    results = await asyncio.gather(*(extract_chunk(llm, c, cfg) for c in chunks))
    facts = [f for group in results for f in group]
    log.info("extracted %d raw facts from %d chunks", len(facts), len(chunks))
    return facts
