"""Trivia-worthiness: document briefs and per-fact trivia scores.

Fact extraction is exhaustive by design (it serves coverage and RAG evaluation), so most facts
are routine details: a simulation temperature, a file format, a run label. Trivia needs the
opposite: the few claims that make a document notable. This module

1. writes a short *brief* per document (summary + the notable nuggets: firsts, records,
   surprising numbers, named contributions, unexpected connections), from the head and tail of
   the document where abstracts and conclusions live; and
2. scores every validated fact for trivia value in the light of that brief.

The scores steer fact sampling (``GenerationContext.fact_weight``) and the brief is shown to the
question writer so it can pick an engaging angle.
"""

from __future__ import annotations

import asyncio
import logging
import re

from pydantic import BaseModel, Field

from trivium.config import TriviaConfig
from trivium.facts.models import Fact
from trivium.facts.store import FactStore
from trivium.ingestion.models import Chunk
from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

# subjects that name the document itself ("Aurora container guide", "Podman documentation")
_DOC_SUBJECT = re.compile(
    r"(?:^|\s)(?:user )?(?:guide|guides|documentation|docs|manual|tutorial|web ?page|page)$", re.I
)

log = logging.getLogger(__name__)


class DocumentBrief(BaseModel):
    document_id: str = ""
    title: str = ""
    summary: str = Field(
        "", description="2-3 sentences: what the document is about and its main result"
    )
    nuggets: list[str] = Field(
        default_factory=list,
        description="the most notable, quotable claims, each one self-contained sentence",
    )


class FactScore(BaseModel):
    fact_id: str
    trivia_score: float = Field(
        ge=0,
        le=1,
        description="0.1 = routine detail (file format, run label, parameter setting); "
        "0.5 = specific, non-obvious fact; 0.9 = a first, record, striking number, key "
        "contribution or surprising connection",
    )


class FactScores(BaseModel):
    scores: list[FactScore] = Field(default_factory=list)


BRIEF_SYSTEM = "You are an editor who finds what makes a document worth quizzing people about."
BRIEF_INSTRUCTIONS = """Read the excerpt (beginning and end of a document) and write a brief:
- `summary`: 2-3 sentences on what the document is about and its main result or contribution.
- `nuggets`: up to {max_nuggets} of the most notable claims a curious, knowledgeable reader would
  enjoy being quizzed on: firsts, records, striking or surprising numbers, named methods, systems or
  people and what they achieved, counter-intuitive findings, unexpected connections. Each nugget is
  one self-contained sentence that names things by their real names (no "the authors", "this work",
  "Table 2", "Run 3"). Skip routine methodological details."""

SCORE_SYSTEM = "You rate facts for how good a trivia question they would make."
SCORE_INSTRUCTIONS = """Rate each fact's `trivia_score` in [0, 1]: how good a trivia question for a knowledgeable
audience could be built on it. Use the document brief to judge what is central or notable.
- 0.9+: a first, a record, a striking number, the key contribution, a surprising connection.
- 0.5: a specific, non-obvious fact with a crisp answer (a named method, person, place, year).
- 0.2: routine detail (a parameter setting, file format, software version, run or sample label).
- 0.0: meaningless without the document (local labels such as "Run 2", "the larger matrix"),
  or about the document itself (what the paper, guide or page covers or describes).
Use the whole scale; most facts in a document are routine. Return one score per `fact_id`."""


def _excerpt(chunks: list[Chunk], max_chars: int) -> str:
    text = "\n\n".join(c.text for c in chunks)
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.65)
    return text[:head] + "\n\n[…]\n\n" + text[-(max_chars - head) :]


async def write_brief(
    llm: LLM, document_id: str, title: str, chunks: list[Chunk], cfg: TriviaConfig
) -> DocumentBrief:
    payload = {"title": title, "excerpt": _excerpt(chunks, cfg.brief_chars)}
    try:
        resp = await llm.generate(
            build_messages(BRIEF_SYSTEM, BRIEF_INSTRUCTIONS.format(max_nuggets=12), payload),
            DocumentBrief,
            task="document_brief",
        )
    except LLMError as exc:
        log.warning("document brief failed for %s: %s", document_id, exc)
        return DocumentBrief(document_id=document_id, title=title)
    brief: DocumentBrief = resp.parsed  # type: ignore[assignment]
    brief.document_id, brief.title = document_id, title
    return brief


SCORE_AUDIENCE = """Audience: rate for the people described in `audience`. A fact scores high when this audience
would want or need to know it (what they must do, use, avoid or expect), not only when it is
remarkable in general."""


async def _score_batch(
    llm: LLM, facts: list[Fact], brief: DocumentBrief, audience: list[str] | None = None
) -> None:
    payload: dict = {
        "document_brief": {"summary": brief.summary, "nuggets": brief.nuggets},
        "facts": [
            {"fact_id": f.id, "statement": f.statement(), "evidence": f.evidence} for f in facts
        ],
    }
    instructions = SCORE_INSTRUCTIONS
    if audience:
        payload["audience"] = audience
        instructions += "\n" + SCORE_AUDIENCE
    try:
        resp = await llm.generate(
            build_messages(SCORE_SYSTEM, instructions, payload),
            FactScores,
            task="score_trivia",
        )
    except LLMError as exc:
        log.warning("trivia scoring failed for %d facts: %s", len(facts), exc)
        return
    by_id = {f.id: f for f in facts}
    for s in resp.parsed.scores:  # type: ignore[union-attr]
        f = by_id.get(s.fact_id)
        if f is not None:
            f.trivia_score = max(0.0, min(1.0, s.trivia_score))
            if _DOC_SUBJECT.search(f.subject.strip()):  # "Aurora container guide | details …"
                f.trivia_score = min(f.trivia_score, 0.05)


async def score_trivia(
    llm: LLM,
    store: FactStore,
    chunks: list[Chunk],
    cfg: TriviaConfig,
    titles: dict[str, str] | None = None,
    audience: list[str] | None = None,
) -> dict[str, DocumentBrief]:
    """Write a brief per document and set ``Fact.trivia_score`` in place; returns the briefs."""
    titles = titles or {}
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.document_id, []).append(c)
    doc_ids = [d for d in store.documents() if d in by_doc]
    briefs = await asyncio.gather(
        *(write_brief(llm, d, titles.get(d, d), by_doc[d], cfg) for d in doc_ids)
    )
    brief_map = {b.document_id: b for b in briefs}
    jobs = []
    for d in doc_ids:
        facts = store.many(store.by_document[d])
        for i in range(0, len(facts), max(1, cfg.score_batch)):
            jobs.append(_score_batch(llm, facts[i : i + cfg.score_batch], brief_map[d], audience))
    await asyncio.gather(*jobs)
    scored = [f.trivia_score for f in store if f.trivia_score is not None]
    if scored:
        log.info(
            "trivia scores: %d/%d facts scored, mean %.2f, %d ≥ 0.7",
            len(scored),
            len(store),
            sum(scored) / len(scored),
            sum(1 for s in scored if s >= 0.7),
        )
    return brief_map
