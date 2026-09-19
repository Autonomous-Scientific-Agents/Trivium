"""Topic → source corpus. MVP backend: Wikipedia (search + plain-text extracts).

Every document records its URL and retrieval timestamp (design §6.1).
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime

from trivium.ingestion.loader import slugify
from trivium.ingestion.models import Document
from trivium.ingestion.parser import parse_text

log = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "trivium-question-generator/0.1 (https://github.com/keceli/trivium)"


def _get(params: dict[str, str]) -> dict:
    url = WIKI_API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — fixed, trusted host
        return json.loads(resp.read().decode("utf-8"))


def _search(topic: str, limit: int) -> list[str]:
    data = _get({"action": "query", "list": "search", "srsearch": topic, "srlimit": str(limit)})
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def _fetch_page(title: str) -> Document | None:
    data = _get(
        {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": "1",
            "inprop": "url",
            "titles": title,
            "redirects": "1",
        }
    )
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract", "")
        if len(text) < 200:
            return None
        doc_id = f"wiki_{slugify(page.get('title', title))}"
        # Wikipedia extracts mark headings as "== Heading =="; convert to markdown for the parser
        md = "\n".join(_heading_to_md(ln) for ln in text.splitlines())
        clean, sections, doc_title = parse_text(
            md, doc_id=doc_id, fallback_title=page.get("title", title)
        )
        return Document(
            id=doc_id,
            title=doc_title,
            source=page.get(
                "fullurl", f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
            ),
            source_type="url",
            retrieved_at=datetime.now(),
            text=clean,
            sections=sections,
        )
    return None


def _heading_to_md(line: str) -> str:
    s = line.strip()
    if s.startswith("==") and s.endswith("=="):
        level = min(6, (len(s) - len(s.lstrip("="))))
        return "#" * level + " " + s.strip("= ").strip()
    return line


async def acquire_topic_sources(topic: str, *, max_sources: int = 5) -> list[Document]:
    loop = asyncio.get_running_loop()
    titles = await loop.run_in_executor(None, _search, topic, max_sources)
    if not titles:
        raise RuntimeError(f"no Wikipedia sources found for topic {topic!r}")
    docs = await asyncio.gather(*(loop.run_in_executor(None, _fetch_page, t) for t in titles))
    out = [d for d in docs if d is not None]
    log.info("topic %r → %d sources: %s", topic, len(out), [d.title for d in out])
    return out
