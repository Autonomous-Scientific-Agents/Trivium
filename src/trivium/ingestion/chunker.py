"""Structure-aware chunking: sections → paragraphs → packed chunks with stable IDs."""

from __future__ import annotations

import re

from trivium.config import ChunkingConfig
from trivium.ingestion.models import Chunk, Document

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def _is_heading_only(paragraph: str, section_title: str) -> bool:
    """A paragraph that is just the section heading carries no facts."""
    if "\n" in paragraph:
        return False
    stripped = paragraph.lstrip("#").strip()
    return paragraph.startswith("#") or stripped.lower() == section_title.lower()


def _split_long(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split ``text`` into pieces of at most ``max_chars`` characters.

    Returns ``(offset, piece)`` pairs where ``offset`` is relative to ``text``, so callers can
    compute exact document offsets without searching. Splits at sentence boundaries; a single
    sentence longer than ``max_chars`` is split at word boundaries.
    """
    if len(text) <= max_chars:
        return [(0, text)]
    # sentence spans (offset, sentence) preserving original positions
    sentences: list[tuple[int, str]] = []
    pos = 0
    for m in _SENT_SPLIT.finditer(text):
        sentences.append((pos, text[pos : m.start()]))
        pos = m.end()
    sentences.append((pos, text[pos:]))
    # break over-long sentences at word boundaries
    units: list[tuple[int, str]] = []
    for off, sent in sentences:
        if len(sent) <= max_chars:
            units.append((off, sent))
            continue
        start = 0
        while start < len(sent):
            end = min(start + max_chars, len(sent))
            if end < len(sent):
                cut = sent.rfind(" ", start, end)
                if cut > start:
                    end = cut
            units.append((off + start, sent[start:end]))
            start = end
    # pack units greedily into pieces that stay within the original text
    out: list[tuple[int, str]] = []
    cur_start: int | None = None
    cur_end = 0
    for off, unit in units:
        if not unit.strip():
            continue
        if cur_start is not None and (off + len(unit)) - cur_start > max_chars:
            out.append((cur_start, text[cur_start:cur_end]))
            cur_start = None
        if cur_start is None:
            cur_start = off
        cur_end = off + len(unit)
    if cur_start is not None:
        out.append((cur_start, text[cur_start:cur_end]))
    return [
        (o, piece.strip()) if piece == piece.strip() else _restrip(o, piece) for o, piece in out
    ]


def _restrip(offset: int, piece: str) -> tuple[int, str]:
    lead = len(piece) - len(piece.lstrip())
    return offset + lead, piece.strip()


def chunk_document(doc: Document, cfg: ChunkingConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkingConfig()
    chunks: list[Chunk] = []
    for sec in doc.sections:
        sec_text = doc.text[sec.start_char : sec.end_char]
        # paragraphs with absolute offsets
        paras: list[tuple[int, int, str]] = []
        pos = 0
        for para in _PARA_SPLIT.split(sec_text):
            idx = sec_text.find(para, pos)
            if idx < 0:
                idx = pos
            pos = idx + len(para)
            lead = len(para) - len(para.lstrip())
            p = para.strip()
            if not p or _is_heading_only(p, sec.title):
                continue
            for rel, piece in _split_long(p, cfg.max_chars):
                start = sec.start_char + idx + lead + rel
                paras.append((start, start + len(piece), piece))

        # pack paragraphs up to target size
        buf: list[tuple[int, int, str]] = []
        buf_len = 0
        n = 0

        def flush(sec=sec) -> None:
            nonlocal buf, buf_len, n
            if not buf:
                return
            text = "\n\n".join(p[2] for p in buf)
            chunks.append(
                Chunk(
                    id=f"{doc.id}_{sec.id}_{n:03d}",
                    document_id=doc.id,
                    section_id=sec.id,
                    section_title=sec.title,
                    text=text,
                    start_char=buf[0][0],
                    end_char=buf[-1][1],
                )
            )
            n += 1
            buf, buf_len = [], 0

        for para in paras:
            if buf and buf_len + len(para[2]) > cfg.target_chars:
                flush()
            buf.append(para)
            buf_len += len(para[2]) + 2
        flush()

    # merge tiny trailing chunks into their predecessor within the same section
    merged: list[Chunk] = []
    for ch in chunks:
        if merged and len(ch.text) < cfg.min_chars and merged[-1].section_id == ch.section_id:
            prev = merged[-1]
            merged[-1] = prev.model_copy(
                update={"text": prev.text + "\n\n" + ch.text, "end_char": ch.end_char}
            )
        else:
            merged.append(ch)
    return merged


def chunk_documents(docs: list[Document], cfg: ChunkingConfig | None = None) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, cfg))
    return out
