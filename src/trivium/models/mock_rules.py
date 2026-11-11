"""Deterministic behaviours for :class:`trivium.models.llm.MockLLM`.

The mock re-implements every LLM task with regular expressions over the JSON payload that
:func:`trivium.prompts.build_messages` embeds in the prompt. It exists so the full adaptive
pipeline can be exercised offline (tests, CI, dry runs). Its "questions" are templated and
its "answers" come from re-parsing the evidence, so accuracy depends on ``skill`` only.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from trivium.facts.models import fmt_number
from trivium.prompts import parse_input

_SUBJ = r"(?P<s>[A-Z][\w\-]*(?: [A-Z0-9][\w\-]*)*)"
_OBJ = r"(?P<o>[^,;]+?)"
_RULES = [
    (
        re.compile(
            _SUBJ
            + r" (?:achieved|obtained|reported|has|had|reached|recorded) (?:an? |the )?(?P<p>[a-zA-Z][a-zA-Z ]+?) of "
            + _OBJ
            + r"\.?$"
        ),
        None,
    ),
    (
        re.compile(
            _SUBJ
            + r" (?:was|is|were|are) (?P<p>(?:founded|born|developed|created|trained|located|used|released|published|introduced|invented|headquartered|based|funded|written|designed) (?:in|by|on|at))(?: the)? "
            + _OBJ
            + r"\.?$"
        ),
        None,
    ),
    (
        re.compile(
            _SUBJ
            + r" (?P<p>uses|used|employs|employed|contains|includes|requires|supports) (?:the |an? )?"
            + _OBJ
            + r"\.?$"
        ),
        None,
    ),
    (
        re.compile(
            r"The (?P<p>[a-z][a-z ]+?) (?:of|for) " + _SUBJ + r" (?:was|is) " + _OBJ + r"\.?$"
        ),
        None,
    ),
    (re.compile(_SUBJ + r" (?P<p>is|was) (?:an? |the )?" + _OBJ + r"\.?$"), None),
]
_NUM = re.compile(r"^([-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?)\s*([A-Za-z%°/µ][\w%°/µ]*(?: [a-z]+)?)?$")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# A tiny "world knowledge" table so closed-book tests can exercise the too-easy rejection path.
WORLD_KNOWLEDGE = {
    "capital of france": "Paris",
    "developed by of xtb": "Grimme Group",
    "developed by of gaussian": "Gaussian Inc",
}


def _snake(p: str) -> str:
    return re.sub(r"\s+", "_", p.strip().lower())


def _words(p: str) -> str:
    return p.replace("_", " ")


def _parse_obj(o: str) -> tuple[float | None, str | None]:
    m = _NUM.match(o.strip())
    if not m:
        return None, None
    try:
        return float(m.group(1).replace(",", "")), (m.group(2) or None)
    except ValueError:
        return None, None


def extract(text: str) -> list[dict[str, Any]]:
    facts = []
    for para in text.split("\n"):
        for sent in _SENT.split(para.strip()):
            sent = sent.strip()
            if not sent or sent.startswith("#"):
                continue
            for rx, _ in _RULES:
                m = rx.match(sent)
                if m:
                    s, p, o = (
                        m.group("s").strip(),
                        _snake(m.group("p")),
                        m.group("o").strip().rstrip("."),
                    )
                    if s.lower() in ("it", "they", "this", "these", "the"):
                        break
                    value, unit = _parse_obj(o)
                    facts.append(
                        {
                            "subject": s,
                            "predicate": p,
                            "object": o,
                            "value": value,
                            "unit": unit,
                            "evidence": sent,
                        }
                    )
                    break
    return facts


def _seeded(*parts: Any) -> float:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _find(facts: list[dict], subject: str, predicate: str) -> dict | None:
    s, p = subject.lower().strip(), _snake(predicate)
    for f in facts:
        if f["subject"].lower() == s and f["predicate"] == p:
            return f
    return None


def _display(f: dict) -> str:
    if f["value"] is not None:
        return (
            f"{fmt_number(f['value'])} {f['unit']}".strip() if f["unit"] else fmt_number(f["value"])
        )
    return f["object"]


# ------------------------------------------------------------------- answering
def answer_from_evidence(question: str, evidence_text: str) -> str | None:
    facts = extract(evidence_text)
    q = question.strip()
    m = re.match(r"True or False: (.+)$", q)
    if m:
        return "True" if statement_label(m.group(1), facts) == "entailed" else "False"
    m = re.match(r"For (?P<a>.+?), what is the (?P<r2>.+?) of its (?P<r1>.+?)\?$", q)
    if m:
        f1 = _find(facts, m.group("a"), m.group("r1"))
        if f1:
            f2 = _find(facts, f1["object"], m.group("r2"))
            return _display(f2) if f2 else None
        return None
    m = re.match(
        r"Among the compared entries, which has the (?P<ext>lowest|highest) (?P<q>.+?)\?$", q
    )
    if m:
        pool = [
            f for f in facts if f["predicate"] == _snake(m.group("q")) and f["value"] is not None
        ]
        if not pool:
            return None
        pick = (
            min(pool, key=lambda f: f["value"])
            if m.group("ext") == "lowest"
            else max(pool, key=lambda f: f["value"])
        )
        return pick["subject"]
    m = re.match(
        r"What is the (?P<op>difference|ratio) of the (?P<q>.+?): (?P<a>.+?) (?P<sym>minus|divided by) (?P<b>.+?)\?$",
        q,
    )
    if m:
        fa, fb = _find(facts, m.group("a"), m.group("q")), _find(facts, m.group("b"), m.group("q"))
        if fa and fb and fa["value"] is not None and fb["value"] is not None:
            if m.group("op") == "difference":
                unit = fa["unit"] or ""
                return f"{fmt_number(fa['value'] - fb['value'])} {unit}".strip()
            return fmt_number(round(fa["value"] / fb["value"], 2)) if fb["value"] else None
        return None
    m = re.match(r"(?:What is the|Which) (?P<rest>.+?)\?$", q)
    if m:
        rest = m.group("rest")
        if " is associated with " in rest:
            p, s = rest.split(" is associated with ", 1)
            f = _find(facts, s, p)
            return _display(f) if f else None
        parts = rest.split(" of ")
        for i in range(1, len(parts)):  # predicate may itself contain "of"
            f = _find(facts, " of ".join(parts[i:]), " of ".join(parts[:i]))
            if f:
                return _display(f)
        return None
    m = re.match(r"What is (?P<s>.+?)\?$", q)
    if m:
        f = _find(facts, m.group("s"), "is")
        return f["object"] if f else None
    return None


_CMP = re.compile(
    _SUBJ
    + r" has an? (?P<rel>higher|lower) (?P<q>[a-z][a-z ]+?) than (?P<b>[A-Z][\w\-]*(?: [A-Z0-9][\w\-]*)*)\.?$"
)


def statement_label(statement: str, facts: list[dict]) -> str:
    m = _CMP.match(statement.strip())
    if m:
        fa, fb = _find(facts, m.group("s"), m.group("q")), _find(facts, m.group("b"), m.group("q"))
        if not (fa and fb) or fa["value"] is None or fb["value"] is None:
            return "neutral"
        higher = fa["value"] > fb["value"]
        return "entailed" if higher == (m.group("rel") == "higher") else "contradicted"
    parsed = extract(statement)
    if not parsed:
        return "neutral"
    st = parsed[0]
    ref = _find(facts, st["subject"], st["predicate"])
    if ref is None:
        return "neutral"
    if st["value"] is not None and ref["value"] is not None:
        return (
            "entailed"
            if abs(st["value"] - ref["value"]) <= 1e-9 * max(1, abs(ref["value"]))
            else "contradicted"
        )
    return (
        "entailed"
        if st["object"].lower().strip() == ref["object"].lower().strip()
        else "contradicted"
    )


def _split_mc(question: str) -> tuple[str, list[str]]:
    lines = question.split("\n")
    opts = [ln[3:].strip() for ln in lines[1:] if re.match(r"^[A-H]\) ", ln)]
    return lines[0], opts


def _verbalize(pred: str, obj: str) -> str:
    p = _words(pred)
    if pred in ("is", "was"):
        return f"{p} {obj}"
    if re.search(r"_(in|by|on|at)$", pred):
        return f"was {p} {obj}"
    if pred in (
        "uses",
        "used",
        "employs",
        "employed",
        "contains",
        "includes",
        "requires",
        "supports",
    ):
        return f"{pred} {obj}"
    return f"has a {p} of {obj}"


# ------------------------------------------------------------------- dispatch
def respond(
    task: str, messages: list[dict[str, str]], *, skill: float, trial: int, model: str
) -> str:
    payload = parse_input(messages)
    j = lambda obj: json.dumps(obj, ensure_ascii=False)  # noqa: E731

    if task == "extract_facts":
        facts = extract(payload.get("text", ""))
        out = [
            {
                **f,
                "qualifiers": {},
                "fact_type": "quantity" if f["value"] is not None else "attribute",
                "explicit": True,
                "confidence": 0.9,
            }
            for f in facts
        ]
        return j({"facts": out})

    if task == "validate_facts":
        js = []
        for f in payload.get("facts", []):
            amb = 0.9 if f["subject"].lower() in ("it", "they", "this") else 0.05
            js.append(
                {
                    "index": f["index"],
                    "entailment": 0.95,
                    "atomicity": 0.95,
                    "explicitness": 1.0,
                    "ambiguity": amb,
                    "notes": "",
                }
            )
        return j({"judgements": js})

    if task == "generate_question":
        facts = payload.get("facts", [])
        strategy = payload.get("strategy", "atomic")
        f0 = facts[0] if facts else {"subject": "X", "predicate": "is", "object": "Y"}
        required = payload.get("required_answer")
        if strategy == "multi_hop" and "A" in payload:
            q = f"For {payload['A']}, what is the {_words(payload['relation2'])} of its {_words(payload['relation1'])}?"
        elif "comparison" in payload:
            q = f"Among the compared entries, which has the {payload['comparison']} {_words(payload['quantity'])}?"
        elif "operation" in payload:
            op = "difference" if " minus " in payload["operation"] else "ratio"
            q = f"What is the {op} of the {_words(f0['predicate'])}: {payload['operation']}?"
        elif strategy == "relationship":
            q = f"Which {_words(f0['predicate'])} is associated with {f0['subject']}?"
        elif f0["predicate"] in ("is", "was"):
            q = f"What is {f0['subject']}?"
        else:
            q = f"What is the {_words(f0['predicate'])} of {f0['subject']}?"
        ans = required if required is not None else f0["object"]
        numeric = bool(re.match(r"^[-+]?\d", str(ans)))
        return j(
            {
                "question": q,
                "answer": str(ans),
                "accepted_answers": [],
                "answer_type": "numeric" if numeric else "short_answer",
                "answer_kind": "number" if numeric else "name",
                "rationale": "mock",
            }
        )

    if task == "write_statement":
        if payload.get("claim") == "comparison":
            return j(
                {
                    "statement": f"{payload['subject']} has a {payload['relation']} "
                    f"{_words(payload['quantity'])} than {payload['other_subject']}."
                }
            )
        return j(
            {
                "statement": f"{payload['subject']} {_verbalize(payload['predicate'], payload['object'])}."
            }
        )

    if task == "falsify_statement":
        obj = str(payload.get("object", ""))
        value, unit = _parse_obj(obj)
        if value is not None:
            wrong = f"{fmt_number(value * 2 if value else 1)} {unit or ''}".strip()
        else:
            wrong = f"Alt-{obj}"
        return j(
            {
                "statement": f"{payload['subject']} {_verbalize(payload['predicate'], wrong)}.",
                "edit": f"{obj} -> {wrong}",
            }
        )

    if task == "document_brief":
        text = payload.get("excerpt", "")
        sents = [s.strip() for s in _SENT.split(text.replace("\n", " ")) if extract(s.strip())]
        return j({"summary": " ".join(sents[:2]), "nuggets": sents[:6]})

    if task == "score_trivia":
        out = []
        for f in payload.get("facts", []):
            st = f.get("statement", "")
            routine = re.search(r"\b(Run \d+|temperature|cutoff|runtime)\b", st)
            out.append(
                {
                    "fact_id": f["fact_id"],
                    "trivia_score": 0.1 if routine else round(0.3 + 0.6 * _seeded("trivia", st), 3),
                }
            )
        return j({"scores": out})

    if task == "options_only":
        opts = payload.get("options", [])
        i = int(_seeded("options_only", model, "|".join(opts), trial) * max(1, len(opts)))
        return j({"answer": "ABCDEFGH"[min(i, max(0, len(opts) - 1))]})

    if task == "pairwise":
        a, b = payload.get("A", {}), payload.get("B", {})
        sa = _seeded("pairwise", a.get("question", ""))
        sb = _seeded("pairwise", b.get("question", ""))
        return j({"better": "A" if sa >= sb else "B", "reason": "mock"})

    if task == "check_statement":
        facts = extract("\n".join(payload.get("evidence", [])))
        return j(
            {
                "label": statement_label(payload.get("statement", ""), facts),
                "confidence": 0.9,
                "explanation": "mock",
            }
        )

    if task == "answer_entailment":
        return j({"entailment": 0.95, "explanation": "mock"})

    if task == "quality":
        q, a = payload.get("question", ""), str(payload.get("reference_answer", ""))
        boolean = a in ("True", "False") and q.startswith("True or False")
        leaked = len(a) >= 3 and a.lower() in q.lower() and not payload.get("options")
        leak = 0.9 if (leaked and not boolean) else 0.05
        concise = 1.0 if len(a.split()) <= 8 else 0.3
        generic = 1.0 if re.search(r"\b(authors?|text|paper|document)\b", q, re.I) else 0.0
        local = bool(re.search(r"\b(Run|Sample|Case) \d+\b|compared entries", q))
        return j(
            {
                "answerability": 0.95,
                "evidence_entailment": 0.95,
                "ambiguity": 0.05,
                "answer_conciseness": concise,
                "clarity": 0.9 - 0.4 * generic,
                "answer_leakage": leak,
                "specificity": 0.9 - 0.5 * generic,
                "interestingness": 0.4 + 0.5 * _seeded("interest", q),
                "standalone": 0.3 if local else 0.95,
                "notes": "mock",
            }
        )

    if task in ("closed_book", "open_book"):
        question = payload.get("question", "")
        base_q, options = _split_mc(question)
        if task == "closed_book":
            key = base_q.lower().rstrip("?")
            ans = next((v for k, v in WORLD_KNOWLEDGE.items() if k in key), None)
            if ans is None:
                ans = "True" if base_q.startswith("True or False") else "unknown"
        else:
            ans = answer_from_evidence(base_q, "\n".join(payload.get("evidence", [])))
            if ans is None:
                ans = "not in evidence"
            elif _seeded(model, base_q, trial) > skill:
                ans = "unknown"  # simulated failure of a weaker model
        if options and ans not in ("unknown", "not in evidence"):
            for i, o in enumerate(options):
                if o.lower().strip() == ans.lower().strip():
                    ans = "ABCDEFGH"[i]
                    break
        return j({"answer": ans})

    if task == "judge_answer":
        pred = str(payload.get("candidate_answer", "")).lower().strip()
        refs = [str(r).lower().strip() for r in payload.get("reference_answers", [])]
        return j({"equivalent": pred in refs, "explanation": "mock"})

    if task == "judge_duplicate":
        return j({"duplicate": False, "explanation": "mock"})

    if task == "distractors":
        correct = str(payload.get("correct_answer", ""))
        pools = payload.get("pools", {})
        out = [
            {"text": v, "kind": kind}
            for kind in ("in_document", "misconception", "near_miss")
            for v in pools.get(kind, [])
            if v.lower() != correct.lower()
        ]
        m = _NUM.match(correct)
        if m:
            v = float(m.group(1).replace(",", ""))
            unit = m.group(2) or ""
            extra = [f"{x:g} {unit}".strip() for x in (v * 0.5, v * 2, v * 1.5, v * 0.8)]
        else:
            # distinct names (variants of the answer itself would be filtered as equivalent)
            extra = ["Orion", "Vega", "Lyra", "Draco", "Cygnus"]
        out += [{"text": x, "kind": "other"} for x in extra]
        return j({"distractors": out[:8]})

    if task == "check_distractor":
        return j({"also_correct": False, "plausible": True, "same_type": True})

    if task == "persona_questions":
        # questions about the outline's sections; grounding (below) decides what is answerable
        sections = [
            (d.get("title", ""), sec) for d in payload.get("outline", []) for sec in d["sections"]
        ]
        asked = set(payload.get("already_asked", []))
        out = []
        gap = "Who do I contact when my allocation runs out?"  # never answerable: a gap report
        if gap not in asked:
            out.append({"question": gap})
        for title, sec in sections:
            q = f"As a newcomer, what should I know about {sec} in {title}?"
            if q not in asked:
                out.append({"question": q})
        return j({"questions": out[:8]})

    if task == "ground_question":
        q = str(payload.get("question", ""))
        for p in payload.get("passages", [])[:1]:
            m = re.search(r"about (.+) in ", q)
            facts = extract(p.get("text", ""))
            if m and facts:
                f = facts[int(_seeded("ground", q) * len(facts))]
                return j(
                    {
                        "status": "answerable",
                        "question": f"What is the {_words(f['predicate'])} of {f['subject']}?",
                        "answer": f["object"],
                        "answer_kind": "number" if f.get("value") is not None else "name",
                        "passage_id": p["passage_id"],
                        "quote": f["evidence"],
                    }
                )
        return j({"status": "unanswerable", "note": "mock: not in the passages"})

    return j({})
