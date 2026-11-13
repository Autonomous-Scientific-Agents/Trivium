"""Cheap, deterministic checks on the *form* of a question and its answer.

These run before any LLM evaluation. They catch the failure modes seen in live runs:
questions that point at the source instead of its content ("…in the Methods section of
doc_2018_…"), questions built on labels that only exist inside the source ("Run 2",
"the larger matrix"), and answers that are not crisp trivia answers ("either method",
"cases where eigenvalues coincide with slice boundaries").
"""

from __future__ import annotations

import re

# References to the source document itself. Rejected in every mode: a RAG benchmark question
# should not depend on knowing the file layout either.
SOURCE_REFERENCE = re.compile(
    r"\bdoc_\w+|\bfact_\w+|\bchunk_\w+"
    r"|\b(?:methods|results|introduction|discussion|conclusions?|abstract|experimental) section\b"
    r"|\b(?:table|figure|fig\.|eq\.|equation|section|appendix|ref\.)\s*\(?[\dIVX]+[a-z]?\)?(?![\w-])"
    r"|\b(?:the|this) (?:text|paper|document|article|passage|excerpt|evidence|manuscript|source)\b"
    r"|\bthe authors?\b"
    # "the Aurora container guide", "this documentation"; a capitalised title ("the Michelin
    # Guide") is a name, not a reference to the source
    r"|\b(?:the|this) (?:[\w-]+ ){0,3}(?-i:guide|guides|documentation|docs|manual|tutorial)\b"
    r"|\bthis (?:web )?page\b"
    r"|\baccording to the (?:text|paper|document|study|article|authors?|source|evidence)\b"
    r"|\bin (?:this|the present|the current) (?:work|study|paper|article)\b"
    r"|\bthe (?:proposed|present|current) (?:method|approach|model|work|study|framework)\b",
    re.I,
)

# Labels that only make sense inside the source. Fine for RAG (the retriever has the source),
# fatal for trivia (the quiz audience does not).
LOCAL_LABEL = re.compile(
    r"\b(?i:run|sample|case|trial|condition|setup|configuration|experiment|scenario|batch|"
    r"participant|specimen)\s+(?:#?\d+|[A-Z]\d*)(?![\w-])"
    r"|\b(?i:the (?:larger|smaller|bigger|first|second|third|last|former|latter|other|new|"
    r"baseline|reference) (?:matrix|matrices|system|systems|model|models|method|methods|run|"
    r"runs|sample|samples|dataset|datasets|one|ones|case|cases|approach|setup|configuration))\b"
    r"|\b(?i:the compared (?:entries|items|methods|models|systems))\b"
)

_GENERIC_WORDS = {
    "either",
    "neither",
    "both",
    "larger",
    "smaller",
    "bigger",
    "higher",
    "lower",
    "more",
    "less",
    "same",
    "different",
    "various",
    "several",
    "none",
    "another",
    "other",
    "former",
    "latter",
    "unknown",
    "it",
    "they",
    "this",
    "that",
    "these",
    "those",
    "important",
    "significant",
}
_CLAUSE_WORDS = {
    "where",
    "when",
    "which",
    "that",
    "because",
    "only",
    "during",
    "while",
    "if",
    "whether",
}
_VERB_START = {
    "includes",
    "including",
    "uses",
    "using",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "can",
    "allows",
    "enables",
    "cases",
    "kept",
    "by",
    "to",
}
_FUNCTION_WORDS = {"of", "the", "in", "at", "to", "for", "by", "with", "on", "from", "a", "an"}
_TOKEN = re.compile(r"[A-Za-z0-9][\w.\-+/%°']*")


def source_reference(question: str) -> str | None:
    m = SOURCE_REFERENCE.search(question)
    return m.group(0) if m else None


def local_label(question: str) -> str | None:
    m = LOCAL_LABEL.search(question)
    return m.group(0) if m else None


def _proper_after_first(tokens: list[str]) -> bool:
    """A capitalised or mixed-case token after the first one (a name, an acronym)."""
    return any(any(ch.isupper() for ch in t) for t in tokens[1:]) or (
        bool(tokens) and sum(ch.isupper() for ch in tokens[0]) >= 2
    )


def generic_answer(answer: str) -> bool:
    """A vague or relative answer ("larger matrix", "either method", "important parameter").

    Names are exempt: "Thomas More" has a capitalised token after the first.
    """
    tokens = _TOKEN.findall(answer)
    if not tokens or any(ch.isdigit() for ch in answer):
        return False
    if _proper_after_first(tokens):
        return False
    return any(t.lower() in _GENERIC_WORDS for t in tokens)


def clause_answer(answer: str) -> bool:
    """A clause or description rather than a short answer ("cases where eigenvalues coincide with
    slice boundaries", "only at the beginning of the SCF cycles", "includes gaps")."""
    tokens = _TOKEN.findall(answer)
    if not tokens or any(ch.isdigit() for ch in answer):
        return False
    low = [t.lower() for t in tokens]
    if low[0] in _VERB_START:
        return True
    if len(tokens) >= 3 and any(t in _CLAUSE_WORDS for t in low):
        return True
    # long, all-lowercase descriptive phrase with function words ("dataset of organic molecules")
    return (
        len(tokens) >= 4
        and not any(any(ch.isupper() for ch in t) for t in tokens)
        and any(t in _FUNCTION_WORDS for t in low)
    )


_STOP = _FUNCTION_WORDS | {
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "when",
    "where",
    "why",
    "does",
    "do",
    "did",
    "can",
    "this",
    "that",
    "it",
    "its",
    "as",
    "into",
}


def content_tokens(text: str) -> list[str]:
    """Lower-cased content words; digit groups are joined ("16,000" → "16000")."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return [
        t for t in (w.lower().strip(".'") for w in _TOKEN.findall(text)) if t and t not in _STOP
    ]


def answer_grounded(answer: str, texts: list[str], min_share: float = 0.5) -> bool:
    """Whether at least ``min_share`` of the answer's content words occur in the evidence.

    Catches answers the writer took from somewhere else than the cited facts (a question about
    Makefiles with evidence about login nodes). Lenient on purpose: "DFT" vs "density functional
    theory (DFT)" passes, and a prefix match tolerates plurals ("Makefile" / "Makefiles")."""
    toks = content_tokens(answer)
    if not toks:
        return True
    have = set(content_tokens(" ".join(texts)))

    def found(t: str) -> bool:
        return t in have or (len(t) >= 5 and any(h.startswith(t[:-1]) for h in have))

    return sum(found(t) for t in toks) / len(toks) >= min_share


def lexical_overlap(question: str, texts: list[str]) -> float | None:
    """Share of the question's content words that also occur in its evidence.

    High values mean the question was written by rephrasing the source (answer leakage, easy
    retrieval); questions real users ask overlap much less with the text that answers them."""
    toks = set(content_tokens(question))
    if not toks:
        return None
    have = set(content_tokens(" ".join(texts)))
    return round(len(toks & have) / len(toks), 4)
