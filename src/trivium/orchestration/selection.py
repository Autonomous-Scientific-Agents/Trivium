"""Best-of selection: rank an accepted pool and keep the top-k (``--best K``).

Acceptance is a threshold test; it says nothing about *which* accepted question is the most
challenging or the most interesting. When the user asks for the best few questions, the
controller generates a larger accepted pool, scores every item with one of the criteria below
and keeps the top-k. All scores are in [0, 1] and are built from the evaluation record only,
so ranking is reproducible from the saved dataset.
"""

from __future__ import annotations

from trivium.config import Difficulty
from trivium.evaluation.evaluator import EvaluatedCandidate

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
    """How good a trivia question it is: the rubric's interestingness, tempered by specificity
    and clarity so that vague-but-flashy items do not win."""
    ev = ec.evaluation
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


def rank(accepted: list[EvaluatedCandidate], by: str) -> list[EvaluatedCandidate]:
    """Return ``accepted`` sorted best-first with ``selection_score``/``selection_rank`` filled in.

    Ties are broken by overall quality, then by reasoning depth, then by candidate id so that
    the order is deterministic for a given run.
    """
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
