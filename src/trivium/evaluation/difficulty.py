"""Empirical difficulty calibration (design §16) with a fallback when no tiers are configured."""

from __future__ import annotations

from trivium.config import Difficulty


def calibrate(
    tier_accuracy: dict[str, float],
    *,
    threshold: float,
    closed_book_accuracy: float | None,
    reasoning_depth: int,
) -> tuple[Difficulty, str]:
    """Return (difficulty, method).

    With small/medium/large accuracies the design thresholds are applied. Otherwise a proxy is used:
    high closed-book accuracy → easy; multi-step reasoning → hard; else medium.
    """
    small = tier_accuracy.get("small")
    medium = tier_accuracy.get("medium")
    large = tier_accuracy.get("large")
    if large is not None and large < threshold:
        return Difficulty.extreme, "tiers"
    if small is not None or medium is not None:
        if small is not None and small >= threshold:
            return Difficulty.easy, "tiers"
        if medium is not None and medium >= threshold:
            return Difficulty.medium if small is not None else Difficulty.easy, "tiers"
        return Difficulty.hard, "tiers"
    # proxy
    if closed_book_accuracy is not None and closed_book_accuracy >= threshold:
        return Difficulty.easy, "proxy"
    if reasoning_depth >= 2:
        return Difficulty.hard, "proxy"
    if closed_book_accuracy is not None and closed_book_accuracy > 0:
        return Difficulty.easy if closed_book_accuracy >= 0.5 else Difficulty.medium, "proxy"
    return Difficulty.medium, "proxy"
