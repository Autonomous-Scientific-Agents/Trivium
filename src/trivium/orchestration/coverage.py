"""Coverage analysis and feedback construction (design §19, §21)."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from trivium.config import GenerationConfig
from trivium.evaluation.evaluator import EvaluatedCandidate
from trivium.facts.store import FactStore, norm
from trivium.generation.base import Feedback


class CoverageReport(BaseModel):
    facts_used: int = 0
    total_valid_facts: int = 0
    fact_coverage: float = 0.0
    documents_used: int = 0
    total_documents: int = 0
    document_coverage: float = 0.0
    sections_used: int = 0
    total_sections: int = 0
    section_coverage: float = 0.0
    entities_used: int = 0
    total_entities: int = 0
    entity_coverage: float = 0.0
    questions_per_section: dict[str, int] = Field(default_factory=dict)
    facts_per_section: dict[str, int] = Field(default_factory=dict)
    difficulty_distribution: dict[str, int] = Field(default_factory=dict)
    type_distribution: dict[str, int] = Field(default_factory=dict)
    strategy_distribution: dict[str, int] = Field(default_factory=dict)
    fact_use_counts: dict[str, int] = Field(default_factory=dict)


def analyze(store: FactStore, accepted: list[EvaluatedCandidate]) -> CoverageReport:
    rep = CoverageReport()
    rep.total_valid_facts = len(store)
    rep.total_documents = len(store.documents())
    sections = store.sections()
    rep.total_sections = len(sections)
    rep.facts_per_section = {f"{d}/{s}": len(ids) for (d, s), ids in store.by_section.items()}
    rep.questions_per_section = {k: 0 for k in rep.facts_per_section}
    entities = store.entities()
    rep.total_entities = len(entities)

    used_facts: Counter[str] = Counter()
    used_docs: set[str] = set()
    used_secs: set[str] = set()
    used_ents: set[str] = set()
    for ec in accepted:
        c = ec.candidate
        for fid in c.source_fact_ids:
            used_facts[fid] += 1
            if fid in store:
                f = store.get(fid)
                used_docs.add(f.document_id)
                key = f"{f.document_id}/{f.section_id}"
                used_secs.add(key)
                rep.questions_per_section[key] = rep.questions_per_section.get(key, 0) + 1
                used_ents.add(norm(f.subject))
        rep.difficulty_distribution[ec.evaluation.difficulty.value] = (
            rep.difficulty_distribution.get(ec.evaluation.difficulty.value, 0) + 1
        )
        rep.type_distribution[c.question_type.value] = (
            rep.type_distribution.get(c.question_type.value, 0) + 1
        )
        rep.strategy_distribution[c.strategy] = rep.strategy_distribution.get(c.strategy, 0) + 1

    rep.facts_used = len(used_facts)
    rep.fact_coverage = (
        round(rep.facts_used / rep.total_valid_facts, 4) if rep.total_valid_facts else 0.0
    )
    rep.documents_used = len(used_docs)
    rep.document_coverage = (
        round(len(used_docs) / rep.total_documents, 4) if rep.total_documents else 0.0
    )
    rep.sections_used = len(used_secs)
    rep.section_coverage = (
        round(len(used_secs) / rep.total_sections, 4) if rep.total_sections else 0.0
    )
    rep.entities_used = len(used_ents & entities)
    rep.entity_coverage = (
        round(rep.entities_used / rep.total_entities, 4) if rep.total_entities else 0.0
    )
    rep.fact_use_counts = dict(used_facts)
    return rep


def build_feedback(
    report: CoverageReport,
    gen_cfg: GenerationConfig,
    rejection_counts: Counter,
    target_total: int,
) -> Feedback:
    fb = Feedback()
    n_accepted = sum(report.strategy_distribution.values())

    # difficulty deficit → target difficulty
    quota = gen_cfg.difficulty_quota()
    if quota:
        deficits = {
            d: q - report.difficulty_distribution.get(d, 0) for d, q in quota.items() if q > 0
        }
        open_ = {d: v for d, v in deficits.items() if v > 0}
        if open_:
            fb.target_difficulty = max(open_, key=lambda d: open_[d] / max(1, quota[d]))
            for d, v in open_.items():
                fb.notes.append(f"{d}: {quota[d] - v}/{quota[d]} accepted so far")

    if fb.target_difficulty is None and gen_cfg.prefer_difficulty:
        fb.target_difficulty = gen_cfg.prefer_difficulty
        if gen_cfg.prefer_difficulty == "hard":
            fb.notes.append(
                "Prefer questions that require combining facts or multi-step reasoning; avoid "
                "single-lookup questions whose answer a well-read person would guess."
            )
    if gen_cfg.select_by in ("trivia", "interest") or (
        gen_cfg.select_top and gen_cfg.select_by is None
    ):
        fb.notes.append(
            "Favour surprising, memorable facts (records, firsts, unusual magnitudes, unexpected "
            "connections between entities) — the kind a good quiz would ask."
        )

    # under-covered sections: many facts, few questions
    ratios = []
    for key, nfacts in report.facts_per_section.items():
        if nfacts < 2:
            continue
        nq = report.questions_per_section.get(key, 0)
        ratios.append((nq / nfacts, -nfacts, key))
    ratios.sort()
    if ratios and n_accepted >= 3:
        worst = [k for r, _, k in ratios[: max(2, len(ratios) // 4)] if r < 0.5]
        fb.priority_sections = [k.split("/", 1) for k in worst]
        zero = [k for r, _, k in ratios if r == 0]
        if zero:
            fb.notes.append(
                "sections with no accepted questions yet: "
                + ", ".join(k.split("/", 1)[1] for k in zero[:6])
            )
        docs_low = sorted({k.split("/", 1)[0] for k in worst})
        fb.priority_documents = docs_low

    # over-used facts
    fb.overused_fact_ids = [fid for fid, n in report.fact_use_counts.items() if n >= 2]
    if fb.overused_fact_ids:
        fb.notes.append("avoid facts already used by several accepted questions")

    # rejection learning (design §22)
    fb.rejection_summary = dict(rejection_counts)
    total_rej = sum(rejection_counts.values())
    if total_rej >= 5:
        top = rejection_counts.most_common(3)
        hints = {
            "duplicate": "previous candidates paraphrased existing questions — target different facts",
            "closed_book_too_easy": "previous candidates were answerable from general knowledge — ask about document-specific values",
            "ambiguous": "previous candidates had several possible answers — add disambiguating context",
            "unsupported_answer": "previous answers were not fully supported by evidence — stay literal to the facts",
            "answer_leaked_in_question": "previous questions contained the answer — avoid restating the object",
            "answer_leakage": "previous questions hinted at the answer — avoid restating the object",
            "large_model_failed_open_book": "previous questions were unanswerable even with evidence — simplify",
            "answer_too_long": f"keep answers to at most {gen_cfg.max_answer_words} words",
            "low_document_dependence": "prefer facts that are specific to these documents",
        }
        for reason, n in top:
            if reason in hints and n / total_rej >= 0.2:
                fb.notes.append(hints[reason])
    return fb
