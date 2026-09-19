from conftest import DATA

from trivium.config import ChunkingConfig
from trivium.ingestion.chunker import chunk_document
from trivium.ingestion.loader import load_inputs
from trivium.ingestion.parser import detect_heading, parse_text


def test_parse_markdown_sections():
    text, sections, title = parse_text(
        "# Title\n\nintro\n\n## Methods\n\nbody\n\n## Results\n\nmore",
        doc_id="d",
        fallback_title="x",
    )
    assert title == "Title"
    assert [s.id for s in sections] == ["title", "methods", "results"]
    # sections tile the text contiguously
    assert sections[0].start_char == 0
    for a, b in zip(sections, sections[1:], strict=False):
        assert a.end_char == b.start_char
    assert sections[-1].end_char == len(text)


def test_detect_headings_variants():
    assert detect_heading("## Results") == (2, "Results")
    assert detect_heading("3.2 Training Details") == (2, "Training Details")
    assert detect_heading("INTRODUCTION") == (1, "Introduction")
    assert detect_heading("This is a normal sentence that ends with a period.") is None


def test_chunker_ids_and_offsets():
    docs = load_inputs([DATA])
    assert {d.id for d in docs} == {"doc_models", "doc_datasets"}
    doc = next(d for d in docs if d.id == "doc_models")
    chunks = chunk_document(doc, ChunkingConfig(target_chars=300, max_chars=500, min_chars=50))
    assert chunks, "no chunks produced"
    assert len({c.id for c in chunks}) == len(chunks)
    for c in chunks:
        assert c.id.startswith(f"{doc.id}_{c.section_id}_")
        # offsets point into the document text
        assert doc.text[c.start_char : c.end_char].strip().startswith(c.text[:20].strip()[:10])
    sections_seen = {c.section_id for c in chunks}
    assert {"introduction", "methods", "results", "conclusion"} <= sections_seen


def test_long_paragraph_is_split():
    long = "Alpha is one. " * 200
    text, sections, _ = parse_text(long, doc_id="d", fallback_title="t")
    from trivium.ingestion.models import Document

    doc = Document(id="d", title="t", source="mem", text=text, sections=sections)
    chunks = chunk_document(doc, ChunkingConfig(target_chars=400, max_chars=600, min_chars=50))
    assert len(chunks) > 3
    assert all(len(c.text) <= 600 + 50 for c in chunks)
