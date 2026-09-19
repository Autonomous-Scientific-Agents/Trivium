import pytest

from trivium.facts.models import Fact, FactObject, looks_unresolved
from trivium.facts.store import FactStore
from trivium.graph.builder import build_graph


def test_knowledge_base_has_expected_facts(knowledge_base):
    docs, chunks, store = knowledge_base
    assert len(docs) == 2 and len(chunks) >= 4
    assert len(store) >= 25
    mace = store.facts_for_subject("MACE")
    preds = {f.predicate for f in mace}
    assert {"mean_absolute_error", "trained_on", "developed_by"} <= preds
    mae = next(f for f in mace if f.predicate == "mean_absolute_error")
    assert mae.object.value == pytest.approx(0.12) and mae.object.unit == "eV"
    assert mae.evidence.startswith("MACE achieved a mean absolute error")
    assert mae.quality is not None and mae.quality.overall >= 0.7
    assert mae.chunk_id.startswith("doc_models_results_")


def test_store_indexes_and_roundtrip(knowledge_base, tmp_path):
    _, _, store = knowledge_base
    assert set(store.documents()) == {"doc_models", "doc_datasets"}
    assert ("doc_models", "results") in store.by_section
    path = tmp_path / "facts.jsonl"
    store.save_jsonl(path)
    again = FactStore.load_jsonl(path)
    assert len(again) == len(store)
    assert again.get(next(iter(store)).id).statement() == next(iter(store)).statement()


def test_unresolved_subjects_flagged():
    assert looks_unresolved("It")
    assert looks_unresolved("the model")
    assert not looks_unresolved("MACE")


def test_graph_has_two_hop_path(knowledge_base):
    _, _, store = knowledge_base
    g = build_graph(store)
    assert g.n_nodes() > 10 and g.n_edges() == len(store)
    paths = g.two_hop_paths()
    chains = {(a.source, a.predicate, b.source, b.predicate, b.target) for a, b in paths}
    assert ("mace", "trained_on", "qm9x", "created_by", "institution y") in chains
    # cross-document path: edge documents differ
    assert any(a.document_id != b.document_id for a, b in paths)


def test_fact_display_formats_numbers():
    f = Fact(
        id="f",
        subject="S",
        predicate="count",
        object=FactObject(text="5000000", value=5_000_000, unit="conformations"),
        document_id="d",
        chunk_id="c",
        evidence="e",
    )
    assert f.object.display() == "5000000 conformations"
    assert FactObject(text="x", value=0.12, unit="eV").display() == "0.12 eV"
