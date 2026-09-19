"""Content-addressed on-disk cache for LLM and embedding calls (SQLite)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def stable_hash(payload: Any) -> str:
    """SHA-256 of a JSON-serialisable payload with sorted keys."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """Tiny thread-safe key/value cache backed by SQLite.

    Keys are hashes of (model, prompt, settings, trial index); values are JSON strings.
    A ``None`` path disables persistence (in-memory only).
    """

    def __init__(self, path: str | Path | None):
        self._lock = threading.Lock()
        self._mem: dict[str, str] = {}
        self._conn: sqlite3.Connection | None = None
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        with self._lock:
            if key in self._mem:
                self.hits += 1
                return self._mem[key]
            if self._conn is not None:
                row = self._conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
                if row:
                    self.hits += 1
                    self._mem[key] = row[0]
                    return row[0]
            self.misses += 1
            return None

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._mem[key] = value
            if self._conn is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
                )
                self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
