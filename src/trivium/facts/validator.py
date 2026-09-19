"""Fact validation: entailment, atomicity, explicitness, ambiguity (design §8)."""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from trivium.config import ExtractionConfig
from trivium.facts.models import Fact, FactQuality
from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)


class FactJudgement(BaseModel):
    index: int
    entailment: float = Field(
        ge=0, le=1, description="1.0 = evidence fully supports the fact; 0.0 = unsupported"
    )
    atomicity: float = Field(
        ge=0, le=1, description="1.0 = a single claim; 0.0 = several unrelated claims"
    )
    explicitness: float = Field(
        ge=0, le=1, description="1.0 = stated verbatim; 0.0 = requires inference"
    )
    ambiguity: float = Field(
        ge=0,
        le=1,
        description="LOWER IS BETTER. 0.0 = subject and object are perfectly clear; 1.0 = hopelessly vague or unresolved",
    )
    notes: str = ""


class ValidationResponse(BaseModel):
    judgements: list[FactJudgement]


VALIDATE_SYSTEM = "You audit extracted facts against their evidence."
VALIDATE_INSTRUCTIONS = """For each fact, score in [0, 1]:
- entailment: the evidence text (and chunk context) clearly supports the fact exactly as stated.
- atomicity: the fact expresses a single claim.
- explicitness: the fact is stated directly rather than inferred.
- ambiguity: subject or object is vague, unresolved, or could refer to several things.
  NOTE THE POLARITY: ambiguity 0.0 = perfectly clear, 1.0 = hopelessly ambiguous. A clear fact gets ~0.05.
Return one judgement per fact, using the same `index` values."""


def overall_quality(q: FactQuality) -> float:
    return round(
        0.5 * q.entailment + 0.2 * q.atomicity + 0.1 * q.explicitness + 0.2 * (1 - q.ambiguity), 4
    )


async def _validate_group(llm: LLM, facts: list[Fact], context: str) -> None:
    payload = {
        "context": context,
        "facts": [
            {
                "index": i,
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object.display(),
                "qualifiers": f.qualifiers,
                "evidence": f.evidence,
            }
            for i, f in enumerate(facts)
        ],
    }
    try:
        resp = await llm.generate(
            build_messages(VALIDATE_SYSTEM, VALIDATE_INSTRUCTIONS, payload),
            ValidationResponse,
            task="validate_facts",
        )
    except LLMError as exc:
        log.warning("fact validation failed for chunk %s: %s", facts[0].chunk_id, exc)
        return
    for j in resp.parsed.judgements:  # type: ignore[union-attr]
        if 0 <= j.index < len(facts):
            q = FactQuality(
                entailment=j.entailment,
                atomicity=j.atomicity,
                explicitness=j.explicitness,
                ambiguity=j.ambiguity,
                notes=j.notes,
            )
            q.overall = overall_quality(q)
            facts[j.index].quality = q


async def validate_facts(
    llm: LLM, facts: list[Fact], chunk_text: dict[str, str], cfg: ExtractionConfig | None = None
) -> list[Fact]:
    """Score facts (grouped per chunk) and drop those under ``cfg.min_quality``."""
    cfg = cfg or ExtractionConfig()
    if not cfg.validate_facts:
        for f in facts:
            f.quality = FactQuality(
                entailment=f.confidence, atomicity=1.0, explicitness=1.0, ambiguity=0.0
            )
            f.quality.overall = overall_quality(f.quality)
        return facts
    by_chunk: dict[str, list[Fact]] = {}
    for f in facts:
        by_chunk.setdefault(f.chunk_id, []).append(f)
    await asyncio.gather(
        *(_validate_group(llm, group, chunk_text.get(cid, "")) for cid, group in by_chunk.items())
    )
    kept = [f for f in facts if f.quality is not None and f.quality.overall >= cfg.min_quality]
    log.info(
        "validated facts: kept %d / %d (min_quality=%.2f)", len(kept), len(facts), cfg.min_quality
    )
    return kept
