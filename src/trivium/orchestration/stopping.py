"""Stopping criteria for the adaptive loop (design §33)."""

from __future__ import annotations

from dataclasses import dataclass, field

from trivium.config import GenerationConfig


@dataclass
class LoopState:
    round: int = 0
    accepted: int = 0
    candidates: int = 0
    rounds_without_acceptance: int = 0
    low_novelty_rounds: int = 0
    difficulty_counts: dict[str, int] = field(default_factory=dict)
    strategies_available: bool = True
    recent_novelty: list[float] = field(default_factory=list)


def should_stop(state: LoopState, cfg: GenerationConfig, target_total: int) -> str | None:
    if state.accepted >= target_total:
        return "quota_reached"
    quota = cfg.difficulty_quota()
    if quota and all(state.difficulty_counts.get(d, 0) >= q for d, q in quota.items()):
        return "difficulty_quotas_reached"
    if not state.strategies_available:
        return "no_applicable_strategies"
    if state.rounds_without_acceptance >= cfg.patience:
        return "exhaustion"
    if state.low_novelty_rounds >= 2:
        return "excessive_redundancy"
    if state.round >= cfg.max_rounds:
        return "max_rounds"
    if cfg.max_candidates is not None and state.candidates >= cfg.max_candidates:
        return "candidate_budget"
    return None


def estimate_target(n_atomic: int, n_relational: int, n_paths: int, max_questions: int) -> int:
    """Design §4 heuristic for the automatic question count."""
    raw = int(0.4 * n_atomic + 0.2 * n_relational + 0.1 * n_paths)
    return max(1, min(max_questions, raw))
