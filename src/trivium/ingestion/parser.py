"""Turn raw text into cleaned text plus a section outline."""

from __future__ import annotations

import re

from trivium.ingestion.models import Section

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,80})$")
_UPPER_HEADING = re.compile(r"^([A-Z][A-Z0-9 \-,:&]{3,60})$")
_KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "method",
    "methodology",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgments",
    "acknowledgements",
    "appendix",
    "related work",
    "experiments",
    "summary",
}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:30] or "section"


def clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line-broken words (PDF)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "vs",
    "with",
}


def _looks_like_title(text: str) -> bool:
    """Section titles are short and mostly capitalised; a wrapped sentence fragment such as
    "The number of eigenvalues in each slice should be close" is not a heading."""
    words = text.split()
    if not words or len(words) > 10:
        return False
    content = [w for w in words if w.lower() not in _TITLE_STOPWORDS]
    if not content:
        return False
    lower = sum(1 for w in content if w[0].islower())
    return lower / len(content) <= 0.5


def _plausible_section_number(num: str) -> bool:
    """ "1806 Journal of …" is a page number, not section 1806."""
    return int(num.split(".")[0]) <= 50


def detect_heading(line: str) -> tuple[int, str] | None:
    s = line.strip()
    if not s or len(s) > 120:
        return None
    m = _MD_HEADING.match(s)
    if m:
        return len(m.group(1)), m.group(2).strip()
    m = _NUMBERED_HEADING.match(s)
    if (
        m
        and not s.endswith(".")
        and _looks_like_title(m.group(2))
        and _plausible_section_number(m.group(1))
    ):
        return m.group(1).count(".") + 1, m.group(2).strip()
    if s.lower().rstrip(":") in _KNOWN_HEADINGS:
        return 1, s.rstrip(":").title()
    m = _UPPER_HEADING.match(s)
    if m and len(s.split()) <= 8:
        return 1, s.title()
    return None


def parse_text(raw: str, *, doc_id: str, fallback_title: str) -> tuple[str, list[Section], str]:
    """Return (clean_text, sections, title). Sections cover the whole text contiguously."""
    text = clean_text(raw)
    lines = text.split("\n")
    headings: list[tuple[int, int, str]] = []  # (char_offset, level, title)
    offset = 0
    in_fence = False
    for ln in lines:
        if ln.lstrip().startswith("```") or ln.lstrip().startswith("~~~"):
            in_fence = not in_fence  # "# comment" inside a code block is not a heading
        h = None if in_fence else detect_heading(ln)
        if h:
            headings.append((offset, h[0], h[1]))
        offset += len(ln) + 1

    title = fallback_title
    if headings and headings[0][1] == 1 and headings[0][0] < 200:
        title = headings[0][2]

    sections: list[Section] = []
    if not headings or headings[0][0] > 0:
        first_end = headings[0][0] if headings else len(text)
        if text[:first_end].strip():
            sections.append(
                Section(id="preamble", title="Preamble", level=1, start_char=0, end_char=first_end)
            )
    used: dict[str, int] = {sec.id: 1 for sec in sections}  # keep "preamble" unique
    for i, (start, level, htitle) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        base = _slug(htitle)
        used[base] = used.get(base, 0) + 1
        sid = base if used[base] == 1 else f"{base}_{used[base]}"
        sections.append(Section(id=sid, title=htitle, level=level, start_char=start, end_char=end))
    if not sections:
        sections.append(Section(id="body", title=title, level=1, start_char=0, end_char=len(text)))
    return text, sections, title
