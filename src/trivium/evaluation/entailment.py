"""Evidence entailment / contradiction checks used by generators and the evaluator."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

log = logging.getLogger(__name__)


class EntailmentVerdict(BaseModel):
    label: Literal["entailed", "contradicted", "neutral"]
    confidence: float = Field(ge=0, le=1, default=0.5)
    explanation: str = ""


class AnswerEntailment(BaseModel):
    entailment: float = Field(ge=0, le=1)
    explanation: str = ""


STATEMENT_SYSTEM = "You are a strict natural-language-inference judge."
STATEMENT_INSTRUCTIONS = """Given the evidence, classify the statement as:
- "entailed": the evidence clearly supports the statement;
- "contradicted": the evidence clearly states something incompatible with the statement;
- "neutral": the evidence neither supports nor contradicts it (e.g. the detail is simply absent).
Absence of information is "neutral", not "contradicted"."""


async def check_statement(llm: LLM, statement: str, evidence: list[str]) -> EntailmentVerdict:
    msgs = build_messages(
        STATEMENT_SYSTEM, STATEMENT_INSTRUCTIONS, {"statement": statement, "evidence": evidence}
    )
    try:
        resp = await llm.generate(msgs, EntailmentVerdict, task="check_statement")
        return resp.parsed  # type: ignore[return-value]
    except LLMError as exc:
        log.warning("statement check failed: %s", exc)
        return EntailmentVerdict(label="neutral", confidence=0.0)


ANSWER_SYSTEM = "You verify that a reference answer follows from evidence."
ANSWER_INSTRUCTIONS = """Score `entailment` in [0, 1]: does the evidence, on its own, imply that the reference answer
is the correct (and only correct) answer to the question? 1.0 = clearly yes; 0.0 = the evidence does
not support the answer or supports a different one."""


async def answer_entailment(
    llm: LLM, question: str, answer: str, evidence: list[str]
) -> AnswerEntailment:
    msgs = build_messages(
        ANSWER_SYSTEM,
        ANSWER_INSTRUCTIONS,
        {"question": question, "reference_answer": answer, "evidence": evidence},
    )
    try:
        resp = await llm.generate(msgs, AnswerEntailment, task="answer_entailment")
        return resp.parsed  # type: ignore[return-value]
    except LLMError as exc:
        log.warning("answer entailment failed: %s", exc)
        return AnswerEntailment(entailment=0.0, explanation=str(exc))
