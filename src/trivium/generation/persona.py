"""Gap-driven questions from a persona (strategy ``persona``).

Fact-driven generators look at a fact and ask what the text plainly answers, so their questions
echo the source wording. This strategy works the other way round:

1. **Ideation.** The persona sees only an outline of the sources (titles, section headings, a
   one-paragraph summary) and asks what it would need to know, one question style at a time
   (factual, procedural, troubleshooting, limits, comparative).
2. **Grounding.** Each question is matched to the passages most similar to it, and a model
   extracts a short answer with a verbatim supporting quote, sharpening the question when needed
   so that it has exactly one answer. The quote must occur in the passage and contain the answer.
3. Questions the sources cannot answer (or only partly) are kept as *gap reports*: they are not
   quiz items, but they show what the documentation does not cover.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from trivium.evaluation.form import answer_grounded
from trivium.facts.models import Fact, FactObject, FactType
from trivium.facts.store import norm
from trivium.generation.base import (
    ANSWER_KINDS,
    PERSONA_QUOTE,
    QUESTION_RULES,
    Candidate,
    GeneratedQuestion,
    Generator,
)
from trivium.models.llm import LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)

STYLES = {
    "factual": "a concrete fact they need (a name, a number, a location, a limit)",
    "procedural": "how to do something; the answer is the command, option, module or setting to use",
    "troubleshooting": "something that went wrong or is confusing; the answer is the cause or fix "
    "(a setting, a command, a component)",
    "limits": "a policy, quota, limit, default or requirement that constrains what they can do",
    "comparative": "a choice between two options (systems, queues, tools, file systems); the answer "
    "is the one to pick or the value that differs",
}


class Idea(BaseModel):
    question: str
    style: str = "factual"


class Ideas(BaseModel):
    questions: list[Idea] = Field(default_factory=list)


class Grounded(BaseModel):
    status: str = Field(description="answerable | partial | unanswerable")
    question: str = ""
    answer: str = ""
    accepted_answers: list[str] = Field(default_factory=list)
    answer_kind: str = "other"
    passage_id: str = ""
    quote: str = ""
    note: str = ""


IDEATE_SYSTEM = "You role-play a real person who needs information from a body of documentation."
IDEATE_INSTRUCTIONS = """You are the person described in `persona`. You have NOT read the sources; you only see their
`outline`. Write {n} questions you would genuinely ask, each in the style `{style}`: {style_hint}.
- Ask from your own goal and gaps in your knowledge, in your own words; do not copy headings.
- Each question must be self-contained and have one short, specific answer (a name, a number with
  unit, a command or option, a setting, a yes/no is NOT allowed).
- Vary the topics across the outline; avoid the questions listed in `already_asked`.
- No generic openers like "What is X?" unless X is something you would really look up."""

GROUND_SYSTEM = "You answer questions strictly from the given passages and quote your evidence."
GROUND_INSTRUCTIONS = """Answer `question` using only `passages`.
- `status`: "answerable" if a passage states the answer explicitly; "partial" if the passages only
  answer part of it or need assumptions; "unanswerable" otherwise.
- If answerable: `answer` is at most {max_words} words (a name, number with unit, command, option,
  setting, term); `quote` is a verbatim sentence or line copied exactly from the passage that
  contains the answer; `passage_id` is that passage's id.
- `question`: the question rewritten, if needed, so that it has exactly one correct answer and makes
  sense without the passages (add the system or context it is about), keeping the asker's voice.
  Do not put the answer in the question.
- Set `answer_kind` as in the rules below. `note`: for partial/unanswerable, what is missing.

{rules}"""


def outline(ctx) -> list[dict[str, Any]]:
    """Per document: title, section headings in order, and the brief's summary when available."""
    docs: dict[str, dict[str, Any]] = {}
    for ch in ctx.chunks.values():
        d = docs.setdefault(
            ch.document_id, {"title": ch.document_id.removeprefix("doc_"), "sections": []}
        )
        title = ch.section_title or ""
        if title and title not in d["sections"]:
            d["sections"].append(title)
    for doc_id, d in docs.items():
        brief = ctx.briefs.get(doc_id)
        if brief is not None:
            d["title"] = brief.title or d["title"]
            if brief.summary:
                d["summary"] = brief.summary
        d["sections"] = d["sections"][:40]
    return list(docs.values())


class PersonaGenerator(Generator):
    name = "persona"

    def __init__(self, ctx):
        super().__init__(ctx)
        self._ideas: list[tuple[str, Idea]] = []  # (persona, idea) not yet grounded
        self._asked: list[str] = []
        self._style_i = 0
        self._vecs: np.ndarray | None = None
        self._vecs_lock = asyncio.Lock()  # propose() grounds ideas concurrently
        self._ids: list[str] = []
        self.gaps: list[dict[str, Any]] = []  # partial / unanswerable questions
        self._n_quotes = 0

    def available(self) -> bool:
        return bool(self.ctx.personas) and bool(self.ctx.chunks) and self.ctx.embedder is not None

    # --------------------------------------------------------------- ideation
    async def _ideate(self, persona: str, style: str) -> list[Idea]:
        n = self.ctx.config.generation.persona_ideas_per_call
        payload = {
            "persona": persona,
            "outline": outline(self.ctx),
            "already_asked": self._asked[-40:],
        }
        text = IDEATE_INSTRUCTIONS.format(n=n, style=style, style_hint=STYLES[style])
        try:
            resp = await self.ctx.llm.generate(
                build_messages(IDEATE_SYSTEM, text, payload),
                Ideas,
                task="persona_questions",
                temperature=0.9,  # ideation is where diversity comes from
                trial=len(self._asked),
            )
        except LLMError as exc:
            log.warning("persona ideation failed: %s", exc)
            return []
        out = []
        for idea in resp.parsed.questions:  # type: ignore[union-attr]
            q = idea.question.strip()
            if q and norm(q) not in {norm(a) for a in self._asked}:
                out.append(Idea(question=q, style=style))
                self._asked.append(q)
        return out

    async def _refill(self, need: int) -> None:
        styles = list(STYLES)
        while len(self._ideas) < need:
            persona = self.ctx.personas[self._style_i % len(self.ctx.personas)]
            style = styles[self._style_i % len(styles)]
            self._style_i += 1
            ideas = await self._ideate(persona, style)
            if not ideas:
                return
            self._ideas.extend((persona, i) for i in ideas)

    # -------------------------------------------------------------- retrieval
    async def _passages(self, question: str, k: int = 4) -> list[dict[str, str]]:
        async with self._vecs_lock:
            if self._vecs is None:
                chunks = list(self.ctx.chunks.values())
                self._ids = [c.id for c in chunks]
                self._vecs = await self.ctx.embedder.embed(
                    [f"{c.section_title}\n{c.text}" for c in chunks]
                )
        q = (await self.ctx.embedder.embed([question]))[0]
        order = np.argsort(-(self._vecs @ q))[:k]
        out = []
        for i in order:
            ch = self.ctx.chunks[self._ids[int(i)]]
            out.append({"passage_id": ch.id, "section": ch.section_title or "", "text": ch.text})
        return out

    # -------------------------------------------------------------- grounding
    async def propose(self, n: int) -> list[Candidate]:
        await self._refill(n)
        picks, self._ideas = self._ideas[:n], self._ideas[n:]
        results = await asyncio.gather(*(self._one(p, idea) for p, idea in picks))
        return [c for c in results if c is not None]

    async def _one(self, persona: str, idea: Idea) -> Candidate | None:
        ctx = self.ctx
        passages = await self._passages(idea.question)
        rules = QUESTION_RULES.format(max_words=ctx.max_answer_words, extra="")
        payload = {"question": idea.question, "passages": passages}
        try:
            resp = await ctx.llm.generate(
                build_messages(
                    GROUND_SYSTEM,
                    GROUND_INSTRUCTIONS.format(max_words=ctx.max_answer_words, rules=rules),
                    payload,
                ),
                Grounded,
                task="ground_question",
            )
        except LLMError as exc:
            log.warning("persona grounding failed: %s", exc)
            return None
        g: Grounded = resp.parsed  # type: ignore[assignment]
        status = g.status.strip().lower()
        if status != "answerable":
            self._gap(idea, persona, status or "unanswerable", g.note)
            return None
        chunk = ctx.chunks.get(g.passage_id)
        quote = g.quote.strip()
        answer = g.answer.strip()
        if (
            chunk is None
            or not quote
            or not answer
            or not _in_text(quote, chunk.text)
            or not answer_grounded(answer, [quote])
            or len(answer.split()) > ctx.max_answer_words
        ):
            self._gap(
                idea, persona, "ungrounded", g.note or "no verbatim quote containing the answer"
            )
            return None
        question = (g.question or idea.question).strip()
        if norm(answer) in norm(question):
            return None  # the sharpened question gives the answer away
        fact = self._quote_fact(chunk, question, answer, quote)
        kind = g.answer_kind if g.answer_kind in (*ANSWER_KINDS, "other") else "other"
        gq = GeneratedQuestion(
            question=question,
            answer=answer,
            accepted_answers=g.accepted_answers,
            answer_type="numeric" if re.match(r"^[-+]?\d", answer) else "short_answer",
            answer_kind=kind,
            rationale=f"persona {idea.style} question",
        )
        return ctx.make_candidate(
            gq,
            [fact],
            self.name,
            derivation={
                "persona": persona,
                "style": idea.style,
                "asked": idea.question,
            },
        )

    def _gap(self, idea: Idea, persona: str, status: str, note: str) -> None:
        self.gaps.append(
            {
                "question": idea.question,
                "style": idea.style,
                "persona": persona,
                "status": status,
                "note": note,
            }
        )

    def _quote_fact(self, chunk, question: str, answer: str, quote: str) -> Fact:
        """A fact holding the supporting quote, so evidence, provenance and coverage work as for
        every other candidate. Marked so that fact-driven strategies do not sample it."""
        self._n_quotes += 1
        fact = Fact(
            id=f"fact_persona_{chunk.id}_{self._n_quotes:03d}",
            subject=question[:200],
            predicate="answered_by",
            object=FactObject(text=answer),
            fact_type=FactType.attribute,
            document_id=chunk.document_id,
            section_id=chunk.section_id,
            chunk_id=chunk.id,
            evidence=quote,
            confidence=0.9,
            derivation=PERSONA_QUOTE,
        )
        self.ctx.store.add(fact)
        return fact


def _in_text(quote: str, text: str) -> bool:
    """Verbatim up to whitespace, case and markdown emphasis/backticks."""

    def squash(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[`*_]", "", s)).strip().lower()

    q = squash(quote).rstrip(".")
    return bool(q) and q in squash(text)
