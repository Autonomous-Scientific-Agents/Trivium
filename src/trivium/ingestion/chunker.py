"""Structure-aware chunking: sections → paragraphs → packed chunks with stable IDs."""

from __future__ import annotations

import re

from trivium.config import ChunkingConfig
from trivium.ingestion.models import Chunk, Document

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    cur = ""
    for sent in _SENT_SPLIT.split(text):
        if cur and len(cur) + len(sent) + 1 > max_chars:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


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
            p = para.strip()
            if not p:
                continue
            for piece in _split_long(p, cfg.max_chars):
                pidx = sec_text.find(piece, idx)
                start = sec.start_char + (pidx if pidx >= 0 else idx)
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
