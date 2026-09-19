"""In-memory fact store with indexes and JSONL persistence (the canonical knowledge source)."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from trivium.facts.models import Fact


class FactStore:
    def __init__(self, facts: Iterable[Fact] = ()):
        self._facts: dict[str, Fact] = {}
        self.by_document: dict[str, list[str]] = defaultdict(list)
        self.by_section: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.by_chunk: dict[str, list[str]] = defaultdict(list)
        self.by_subject: dict[str, list[str]] = defaultdict(list)
        self.by_predicate: dict[str, list[str]] = defaultdict(list)
        for f in facts:
            self.add(f)

    # ------------------------------------------------------------ mutation
    def add(self, fact: Fact) -> None:
        if fact.id in self._facts:
            return
        self._facts[fact.id] = fact
        self.by_document[fact.document_id].append(fact.id)
        self.by_section[(fact.document_id, fact.section_id)].append(fact.id)
        self.by_chunk[fact.chunk_id].append(fact.id)
        self.by_subject[norm(fact.subject)].append(fact.id)
        self.by_predicate[fact.predicate].append(fact.id)

    def extend(self, facts: Iterable[Fact]) -> None:
        for f in facts:
            self.add(f)

    # -------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self._facts.values())

    def __contains__(self, fact_id: str) -> bool:
        return fact_id in self._facts

    def get(self, fact_id: str) -> Fact:
        return self._facts[fact_id]

    def many(self, ids: Iterable[str]) -> list[Fact]:
        return [self._facts[i] for i in ids if i in self._facts]

    def all(self) -> list[Fact]:
        return list(self._facts.values())

    def facts_for_subject(self, subject: str) -> list[Fact]:
        return self.many(self.by_subject.get(norm(subject), []))

    def facts_for_predicate(self, predicate: str) -> list[Fact]:
        return self.many(self.by_predicate.get(predicate, []))

    def numeric_facts(self) -> list[Fact]:
        return [f for f in self if f.object.is_numeric]

    def documents(self) -> list[str]:
        return list(self.by_document)

    def sections(self) -> list[tuple[str, str]]:
        return list(self.by_section)

    def entities(self) -> set[str]:
        ents = {norm(f.subject) for f in self}
        ents |= {
            norm(f.object.text) for f in self if not f.object.is_numeric and len(f.object.text) < 60
        }
        return ents

    # --------------------------------------------------------- persistence
    def save_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for f in self:
                fh.write(f.model_dump_json() + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> FactStore:
        store = cls()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    store.add(Fact.model_validate(json.loads(line)))
        return store


def norm(s: str) -> str:
    return " ".join(s.lower().split())
