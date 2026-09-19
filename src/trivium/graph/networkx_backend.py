from __future__ import annotations

import networkx as nx

from trivium.graph.backend import Edge


class NetworkXGraph:
    """MultiDiGraph where every edge carries the id of the fact it came from."""

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    def add_node(self, node: str, **attrs: object) -> None:
        self.g.add_node(node, **attrs)

    def add_edge(
        self, source: str, target: str, *, predicate: str, fact_id: str, document_id: str
    ) -> None:
        self.g.add_edge(
            source,
            target,
            key=fact_id,
            predicate=predicate,
            fact_id=fact_id,
            document_id=document_id,
        )

    def nodes(self) -> list[str]:
        return list(self.g.nodes)

    def _edge(self, u: str, v: str, d: dict) -> Edge:
        return Edge(
            source=u,
            target=v,
            predicate=d["predicate"],
            fact_id=d["fact_id"],
            document_id=d["document_id"],
        )

    def edges(self) -> list[Edge]:
        return [self._edge(u, v, d) for u, v, d in self.g.edges(data=True)]

    def out_edges(self, node: str) -> list[Edge]:
        if node not in self.g:
            return []
        return [self._edge(u, v, d) for u, v, d in self.g.out_edges(node, data=True)]

    def in_edges(self, node: str) -> list[Edge]:
        if node not in self.g:
            return []
        return [self._edge(u, v, d) for u, v, d in self.g.in_edges(node, data=True)]

    def degree(self, node: str) -> int:
        return self.g.degree(node) if node in self.g else 0

    def two_hop_paths(self, max_paths: int = 10_000) -> list[tuple[Edge, Edge]]:
        """A -e1-> B -e2-> C with A != C and B an entity (not a literal) node."""
        paths: list[tuple[Edge, Edge]] = []
        for b in self.g.nodes:
            if self.g.nodes[b].get("kind") == "literal":
                continue
            ins = self.in_edges(b)
            outs = self.out_edges(b)
            if not ins or not outs:
                continue
            for e1 in ins:
                for e2 in outs:
                    if e1.source == e2.target or e1.fact_id == e2.fact_id:
                        continue
                    paths.append((e1, e2))
                    if len(paths) >= max_paths:
                        return paths
        return paths

    def n_nodes(self) -> int:
        return self.g.number_of_nodes()

    def n_edges(self) -> int:
        return self.g.number_of_edges()
