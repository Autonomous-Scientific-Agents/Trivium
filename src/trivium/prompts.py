"""Prompt construction helpers shared by all LLM-facing components.

Every prompt is a system instruction plus a user message that ends with a fenced JSON
block labelled ``INPUT``. Keeping the machine-readable input in one place makes prompts
easy to version, cache and simulate (the mock backend parses the same block).
"""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "2026.09.1"

SYSTEM_BASE = (
    "You are a meticulous scientific assistant helping build an objectively gradable "
    "question bank from source documents. Follow the output format exactly."
)


def build_messages(system: str, instructions: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    body = (
        instructions.rstrip()
        + "\n\nINPUT:\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=1)
        + "\n```"
    )
    return [
        {"role": "system", "content": f"{SYSTEM_BASE}\n\n{system}".strip()},
        {"role": "user", "content": body},
    ]


def parse_input(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Recover the INPUT payload from messages built by :func:`build_messages`."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        marker = "INPUT:\n```json\n"
        i = m["content"].find(marker)
        if i == -1:
            continue
        blob = m["content"][i + len(marker) :]
        end = blob.rfind("```")
        try:
            return json.loads(blob[:end] if end != -1 else blob)
        except json.JSONDecodeError:
            return {}
    return {}
