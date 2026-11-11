"""Feedback report: what reviewers' thumbs say about the generator and the automatic scores.

``trivium feedback-report`` reads the ``feedback.json`` files written by the review UI and reports
the share of questions rated 👍, broken down by strategy, type and persona style, and how well
each automatic score agrees with the reviewers (AUC: the probability that a random 👍 question
scores higher than a random 👎 one; 0.5 = no better than chance).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from trivium.web.store import Workspace, display_answer

SCORES = (
    "overall_quality",
    "interestingness",
    "standalone",
    "selection_score",
    "pairwise_win_rate",
    "document_dependence",
    "closed_book_accuracy",
    "novelty",
    "lexical_overlap",
)


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def _rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    up = sum(r["rating"] == "up" for r in rows)
    down = sum(r["rating"] == "down" for r in rows)
    return {
        "rated": up + down,
        "up": up,
        "down": down,
        "up_rate": round(up / (up + down), 3) if up + down else None,
    }


def collect(workspace: str | Path, runs: list[str] | None = None) -> list[dict[str, Any]]:
    """One row per rated (or commented) question across the selected runs."""
    ws = Workspace(workspace)
    ids = runs or [r["id"] for r in ws.list_runs()]
    rows: list[dict[str, Any]] = []
    for run_id in ids:
        d = ws.run_dir(run_id)
        data = ws.load_dataset(d) or {}
        fb = ws.feedback(d)
        for q in data.get("questions", []):
            f = fb.get(q.get("id"))
            if not f:
                continue
            gen = q.get("generation") or {}
            deriv = gen.get("derivation") or {}
            rows.append(
                {
                    "run": run_id,
                    "id": q.get("id"),
                    "question": q.get("question"),
                    "answer": display_answer(q),
                    "type": q.get("type"),
                    "strategy": gen.get("strategy"),
                    "style": deriv.get("style"),
                    "difficulty": q.get("difficulty"),
                    "rating": f.get("rating"),
                    "comment": f.get("comment", ""),
                    "scores": {k: (q.get("evaluation") or {}).get(k) for k in SCORES},
                }
            )
    return rows


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rated = [r for r in rows if r["rating"] in ("up", "down")]
    by: dict[str, dict[str, list]] = {
        k: defaultdict(list) for k in ("run", "type", "strategy", "style")
    }
    for r in rated:
        for k in by:
            if r.get(k):
                by[k][r[k]].append(r)
    agreement = {}
    for s in SCORES:
        pos = [
            r["scores"].get(s)
            for r in rated
            if r["rating"] == "up" and r["scores"].get(s) is not None
        ]
        neg = [
            r["scores"].get(s)
            for r in rated
            if r["rating"] == "down" and r["scores"].get(s) is not None
        ]
        a = auc(pos, neg)
        if a is not None:
            agreement[s] = {
                "auc": a,
                "mean_up": round(sum(pos) / len(pos), 3),
                "mean_down": round(sum(neg) / len(neg), 3),
            }
    return {
        "overall": _rate(rated),
        **{f"by_{k}": {v: _rate(rs) for v, rs in sorted(g.items())} for k, g in by.items()},
        "score_agreement": agreement,
        "down_comments": [
            {
                "run": r["run"],
                "question": r["question"],
                "answer": r["answer"],
                "comment": r["comment"],
            }
            for r in rated
            if r["rating"] == "down" and r["comment"]
        ],
    }
