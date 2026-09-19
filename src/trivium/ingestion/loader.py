"""Discover and load input documents (.txt, .md, .pdf, .html, directories)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from trivium.ingestion.models import Document
from trivium.ingestion.parser import parse_text

log = logging.getLogger(__name__)

SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".html", ".htm", ".rst"}


def discover_files(inputs: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            files.extend(
                sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED and f.is_file())
            )
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED:
                log.warning("skipping unsupported file type: %s", p)
                continue
            files.append(p)
        else:
            raise FileNotFoundError(f"input not found: {p}")
    return files


def slugify(name: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug[:max_len] or "doc"


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install 'trivium[pdf]' to read PDF files") from exc
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return _strip_repeated_lines(pages)


def _strip_repeated_lines(pages: list[str]) -> str:
    """Remove running headers/footers: lines that repeat on most pages."""
    if len(pages) < 3:
        return "\n\n".join(pages)
    from collections import Counter

    counts: Counter[str] = Counter()
    for page in pages:
        lines = {ln.strip() for ln in page.splitlines() if ln.strip()}
        counts.update(lines)
    threshold = max(3, int(0.6 * len(pages)))
    noise = {ln for ln, c in counts.items() if c >= threshold and len(ln) < 120}
    cleaned = []
    for page in pages:
        keep = [
            ln
            for ln in page.splitlines()
            if ln.strip() not in noise and not re.fullmatch(r"\s*\d+\s*", ln)
        ]
        cleaned.append("\n".join(keep))
    return "\n\n".join(cleaned)


def read_html(path: Path) -> str:
    from html.parser import HTMLParser

    class _Text(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip += 1
            if tag in ("h1", "h2", "h3"):
                self.parts.append("\n" + "#" * int(tag[1]) + " ")
            elif tag in ("p", "br", "div", "li", "tr"):
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self._skip = max(0, self._skip - 1)
            if tag in ("h1", "h2", "h3", "p"):
                self.parts.append("\n")

        def handle_data(self, data):
            if not self._skip:
                self.parts.append(data)

    p = _Text()
    p.feed(path.read_text(encoding="utf-8", errors="replace"))
    return re.sub(r"\n{3,}", "\n\n", "".join(p.parts))


def load_file(path: Path, doc_id: str | None = None) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = read_pdf(path)
    elif suffix in (".html", ".htm"):
        raw = read_html(path)
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
    doc_id = doc_id or f"doc_{slugify(path.stem)}"
    text, sections, title = parse_text(raw, doc_id=doc_id, fallback_title=path.stem)
    return Document(
        id=doc_id, title=title, source=str(path), source_type="file", text=text, sections=sections
    )


def load_inputs(inputs: list[str | Path]) -> list[Document]:
    files = discover_files(inputs)
    if not files:
        raise FileNotFoundError(f"no supported documents found under {inputs}")
    docs: list[Document] = []
    seen: dict[str, int] = {}
    for f in files:
        base = f"doc_{slugify(f.stem)}"
        seen[base] = seen.get(base, 0) + 1
        doc_id = base if seen[base] == 1 else f"{base}_{seen[base]}"
        doc = load_file(f, doc_id)
        if len(doc.text.strip()) < 50:
            log.warning("document %s is nearly empty, skipping", f)
            continue
        docs.append(doc)
        log.info("loaded %s (%d chars, %d sections)", f.name, doc.n_chars, len(doc.sections))
    return docs
