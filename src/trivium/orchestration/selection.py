"""Best-of selection: rank an accepted pool and keep the top-k (``--best K``).

Acceptance is a threshold test; it says nothing about *which* accepted question is the most
challenging or the most interesting. When the user asks for the best few questions, the
controller generates a larger accepted pool, scores every item with one of the criteria below
and keeps the top-k. All scores are in [0, 1] and are built from the evaluation record only,
so ranking is reproducible from the saved dataset.
"""

from __future__ import annotations

import asyncio
import random

from pydantic import BaseModel

from trivium.config import Difficulty
from trivium.evaluation.evaluator import EvaluatedCandidate
from trivium.models.llm import LLM, LLMError
from trivium.prompts import build_messages

SELECT_BY = ("challenge", "interest", "trivia", "quality")

_DIFFICULTY_RANK = {
    Difficulty.extreme: 1.0,
    Difficulty.hard: 0.75,
    Difficulty.medium: 0.4,
    Difficulty.easy: 0.1,
    Difficulty.unknown: 0.3,
}


def challenge_score(ec: EvaluatedCandidate) -> float:
    """How hard the question is for models: difficulty label, weaker-tier failure, closed-book
    failure and reasoning depth."""
    ev, cand = ec.evaluation, ec.candidate
    label = _DIFFICULTY_RANK.get(ev.difficulty, 0.3)
    weak = [ev.tier_accuracy[t] for t in ("small", "medium") if t in ev.tier_accuracy]
    if weak:
        tier_fail = 1.0 - sum(weak) / len(weak)
    else:  # no tiers configured: fall back to the closed-book signal only
        tier_fail = 1.0 - (ev.closed_book_chance_corrected or 0.0)
    closed_fail = 1.0 - (
        ev.closed_book_chance_corrected
        if ev.closed_book_chance_corrected is not None
        else (ev.closed_book_accuracy or 0.0)
    )
    depth = min(cand.reasoning_depth, 3) / 3.0
    return round(0.4 * label + 0.3 * tier_fail + 0.2 * closed_fail + 0.1 * depth, 4)


def interest_score(ec: EvaluatedCandidate) -> float:
    """How good a trivia question it is: the rubric's interestingness (or, when a pairwise
    tournament was run, the win rate, which discriminates far better than absolute scores that
    cluster around 0.7), tempered by specificity and clarity so vague-but-flashy items do not win."""
    ev = ec.evaluation
    if ev.pairwise_win_rate is not None:
        interest = ev.pairwise_win_rate
    else:
        interest = ev.interestingness if ev.interestingness is not None else 0.5
    specificity = ev.specificity if ev.specificity is not None else 0.5
    clarity = ev.clarity if ev.clarity is not None else 0.5
    return round(0.7 * interest + 0.15 * specificity + 0.15 * clarity, 4)


def selection_score(ec: EvaluatedCandidate, by: str) -> float:
    if by == "challenge":
        return challenge_score(ec)
    if by == "interest":
        return interest_score(ec)
    if by == "trivia":  # hard *and* interesting
        return round(0.55 * challenge_score(ec) + 0.45 * interest_score(ec), 4)
    if by == "quality":
        return round(ec.evaluation.overall_quality or 0.0, 4)
    raise ValueError(f"unknown selection criterion {by!r}; expected one of {SELECT_BY}")


def rank(
    accepted: list[EvaluatedCandidate], by: str, *, pairwise: dict[str, float] | None = None
) -> list[EvaluatedCandidate]:
    """Return ``accepted`` sorted best-first with ``selection_score``/``selection_rank`` filled in.

    ``pairwise`` (candidate id → tournament win rate) replaces the rubric interestingness.
    Ties are broken by overall quality, then by reasoning depth, then by candidate id so that
    the order is deterministic for a given run.
    """
    if pairwise:
        for ec in accepted:
            if ec.candidate.id in pairwise:
                ec.evaluation.pairwise_win_rate = round(pairwise[ec.candidate.id], 4)
    for ec in accepted:
        ec.evaluation.selection_score = selection_score(ec, by)
    ordered = sorted(
        accepted,
        key=lambda ec: (
            -(ec.evaluation.selection_score or 0.0),
            -(ec.evaluation.overall_quality or 0.0),
            -ec.candidate.reasoning_depth,
            ec.candidate.id,
        ),
    )
    for i, ec in enumerate(ordered):
        ec.evaluation.selection_rank = i + 1
    return ordered


# ------------------------------------------------------------------ tournament
class PairwiseVerdict(BaseModel):
    better: str  # "A" or "B"
    reason: str = ""


PAIRWISE_SYSTEM = "You are a quizmaster choosing questions for a knowledgeable trivia audience."
PAIRWISE_INSTRUCTIONS = """Two candidate quiz questions are shown with their answers. Which one is the better trivia
question? Prefer the one that is more interesting or surprising to learn, has a crisper answer, makes
sense on its own, and rewards knowledge rather than lookup of a routine detail. Answer `better` with
"A" or "B"."""


def _display(ec: EvaluatedCandidate) -> dict:
    c = ec.candidate
    answer = c.answer
    if c.options and c.answer_type == "choice":
        answer = (c.derivation or {}).get("base_answer", answer)
    return {"question": c.question, "options": c.options, "answer": answer}


PAIRWISE_AUDIENCE = """The questions are for the audience described in `audience`: prefer the one this audience
would find more useful or more rewarding to know, phrased as they would ask it."""


async def _duel(
    judge: LLM,
    a: EvaluatedCandidate,
    b: EvaluatedCandidate,
    swap: bool,
    audience: list[str] | None = None,
) -> EvaluatedCandidate | None:
    first, second = (b, a) if swap else (a, b)
    payload: dict = {"A": _display(first), "B": _display(second)}
    instructions = PAIRWISE_INSTRUCTIONS
    if audience:
        payload["audience"] = audience
        instructions += "\n" + PAIRWISE_AUDIENCE
    try:
        resp = await judge.generate(
            build_messages(PAIRWISE_SYSTEM, instructions, payload),
            PairwiseVerdict,
            task="pairwise",
        )
    except LLMError:
        return None
    pick = resp.parsed.better.strip()[:1].upper()  # type: ignore[union-attr]
    if pick == "A":
        return first
    if pick == "B":
        return second
    return None


async def tournament(
    judge: LLM,
    pool: list[EvaluatedCandidate],
    *,
    rounds: int = 3,
    rng: random.Random | None = None,
    audience: list[str] | None = None,
) -> dict[str, float]:
    """Swiss-style pairwise tournament for "which is the better trivia question".

    Each round pairs questions with similar records (ties broken randomly); the A/B order is
    randomised to cancel position bias. Returns candidate id → win rate (0.5 when unplayed).
    Cost: ``rounds * len(pool) / 2`` judge calls.
    """
    rng = rng or random.Random(0)
    wins = {ec.candidate.id: 0 for ec in pool}
    games = {ec.candidate.id: 0 for ec in pool}
    if len(pool) < 2:
        return {k: 0.5 for k in wins}
    for _ in range(rounds):
        order = list(pool)
        rng.shuffle(order)
        order.sort(key=lambda ec: -wins[ec.candidate.id])  # stable: shuffled within equal records
        pairs = [(order[i], order[i + 1]) for i in range(0, len(order) - 1, 2)]
        results = await asyncio.gather(
            *(_duel(judge, a, b, rng.random() < 0.5, audience) for a, b in pairs)
        )
        for (a, b), winner in zip(pairs, results, strict=True):
            if winner is None:
                continue
            games[a.candidate.id] += 1
            games[b.candidate.id] += 1
            wins[winner.candidate.id] += 1
    return {k: (wins[k] / games[k] if games[k] else 0.5) for k in wins}
