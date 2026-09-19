"""Backend-independent knowledge-graph interface (design §9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    predicate: str
    fact_id: str
    document_id: str


@runtime_checkable
class GraphBackend(Protocol):
    def add_node(self, node: str, **attrs: object) -> None: ...

    def add_edge(
        self, source: str, target: str, *, predicate: str, fact_id: str, document_id: str
    ) -> None: ...

    def nodes(self) -> list[str]: ...

    def edges(self) -> list[Edge]: ...

    def out_edges(self, node: str) -> list[Edge]: ...

    def in_edges(self, node: str) -> list[Edge]: ...

    def degree(self, node: str) -> int: ...

    def two_hop_paths(self, max_paths: int = 10_000) -> list[tuple[Edge, Edge]]: ...

    def n_nodes(self) -> int: ...

    def n_edges(self) -> int: ...
