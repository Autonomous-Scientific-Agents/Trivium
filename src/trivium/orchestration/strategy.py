"""Heuristic, interpretable strategy weighting (design §22–23)."""

from __future__ import annotations

from dataclasses import dataclass, field

from trivium.config import GenerationConfig, QuestionType
from trivium.orchestration.coverage import CoverageReport


@dataclass
class StrategyStats:
    generated: int = 0
    evaluated: int = 0
    accepted: int = 0
    hard: int = 0
    duplicates: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted + 1) / (self.evaluated + 2)  # Laplace-smoothed

    def as_dict(self) -> dict:
        return {
            "generated": self.generated,
            "evaluated": self.evaluated,
            "accepted": self.accepted,
            "hard_accepted": self.hard,
            "duplicates": self.duplicates,
            "acceptance_rate": round(self.accepted / self.evaluated, 4) if self.evaluated else None,
            "rejections": dict(self.rejections),
        }


class StrategySelector:
    def __init__(self, gen_cfg: GenerationConfig, available: dict[str, bool]):
        self.cfg = gen_cfg
        self.available = available
        self.stats: dict[str, StrategyStats] = {s: StrategyStats() for s in available}
        self.base = dict(gen_cfg.strategy_weights)
        if QuestionType.true_false in gen_cfg.question_types:
            self.base.setdefault("true_false", 0.2)

    # ------------------------------------------------------------- bookkeeping
    def record_generated(self, strategy: str, n: int) -> None:
        self.stats.setdefault(strategy, StrategyStats()).generated += n

    def record_result(self, strategy: str, accepted: bool, reasons: list[str], hard: bool) -> None:
        st = self.stats.setdefault(strategy, StrategyStats())
        st.evaluated += 1
        if accepted:
            st.accepted += 1
            if hard:
                st.hard += 1
        for r in reasons:
            st.rejections[r] = st.rejections.get(r, 0) + 1
            if r == "duplicate":
                st.duplicates += 1

    # ---------------------------------------------------------------- weights
    def weights(
        self, report: CoverageReport, target_difficulty: str | None, target_total: int
    ) -> dict[str, float]:
        n_acc = max(1, sum(report.strategy_distribution.values()))
        rates = {s: st.acceptance_rate for s, st in self.stats.items() if st.evaluated}
        mean_rate = sum(rates.values()) / len(rates) if rates else None
        w: dict[str, float] = {}
        for s, base in self.base.items():
            if not self.available.get(s, False):
                continue
            x = base
            st = self.stats.get(s)
            if st and st.evaluated >= 4 and mean_rate:
                x *= max(
                    0.2, min(3.0, st.acceptance_rate / mean_rate)
                )  # W_s ∝ accepted / generated
                if st.evaluated >= 6 and st.accepted == 0:
                    x *= 0.3
            if target_difficulty == "hard":
                x *= {
                    "atomic": 0.4,
                    "true_false": 0.5,
                    "comparison": 1.6,
                    "derived": 1.6,
                    "multi_hop": 1.8,
                    "cross_document": 1.5,
                }.get(s, 1.0)
                if st and st.accepted and st.hard / st.accepted > 0.5:
                    x *= 1.5
            elif target_difficulty == "easy":
                x *= {
                    "atomic": 1.6,
                    "relationship": 1.3,
                    "true_false": 1.3,
                    "multi_hop": 0.5,
                    "derived": 0.5,
                }.get(s, 1.0)
            w[s] = x
        # ratio targets for multi-hop / cross-document
        for s, ratio in (
            ("multi_hop", self.cfg.multi_hop_ratio),
            ("cross_document", self.cfg.cross_document_ratio),
        ):
            if s in w and ratio > 0:
                share = report.strategy_distribution.get(s, 0) / n_acc
                if share < ratio:
                    w[s] *= 1.0 + 2.0 * (ratio - share) / ratio
                elif share > 1.5 * ratio:
                    w[s] *= 0.4
        # true/false share follows the requested type ratio
        if "true_false" in w:
            want = self._type_target("true_false")
            share = report.type_distribution.get("true_false", 0) / n_acc
            w["true_false"] *= 2.0 if share < want else (0.3 if share > 1.3 * want else 1.0)
        total = sum(w.values())
        return {s: v / total for s, v in w.items()} if total else {}

    def _type_target(self, qtype: str) -> float:
        types = [t.value for t in self.cfg.question_types]
        if self.cfg.type_ratio and qtype in self.cfg.type_ratio:
            return self.cfg.type_ratio[qtype]
        return 1.0 / len(types) if qtype in types else 0.0

    def allocate(self, batch_size: int, weights: dict[str, float], rng) -> dict[str, int]:
        """Split a batch across strategies proportionally, with stochastic rounding."""
        if not weights:
            return {}
        alloc: dict[str, int] = {}
        remaining = batch_size
        items = sorted(weights.items(), key=lambda kv: -kv[1])
        for s, wgt in items:
            n = int(wgt * batch_size)
            alloc[s] = n
            remaining -= n
        strategies, probs = zip(*items, strict=True)
        for _ in range(max(0, remaining)):
            s = rng.choices(strategies, weights=probs, k=1)[0]
            alloc[s] += 1
        return {s: n for s, n in alloc.items() if n > 0}

    def summary(self) -> dict[str, dict]:
        return {s: st.as_dict() for s, st in self.stats.items() if st.generated}
