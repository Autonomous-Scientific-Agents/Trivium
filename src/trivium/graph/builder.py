"""Build a provenance-preserving knowledge graph from validated facts."""

from __future__ import annotations

from trivium.facts.store import FactStore, norm
from trivium.graph.backend import GraphBackend
from trivium.graph.networkx_backend import NetworkXGraph


def build_graph(store: FactStore, backend: GraphBackend | None = None) -> GraphBackend:
    g: GraphBackend = backend or NetworkXGraph()
    subjects = {norm(f.subject) for f in store}
    for f in store:
        s = norm(f.subject)
        g.add_node(s, kind="entity", label=f.subject)
        if f.object.is_numeric:
            o = f"{s}::{f.predicate}::{f.object.display()}"
            g.add_node(
                o,
                kind="literal",
                label=f.object.display(),
                value=f.object.value,
                unit=f.object.unit,
            )
        else:
            o = norm(f.object.text)
            # an object that is also a subject somewhere is an entity → enables multi-hop paths
            kind = "entity" if (o in subjects or len(o.split()) <= 4) else "literal"
            g.add_node(o, kind=kind, label=f.object.text)
        g.add_edge(s, o, predicate=f.predicate, fact_id=f.id, document_id=f.document_id)
    return g
