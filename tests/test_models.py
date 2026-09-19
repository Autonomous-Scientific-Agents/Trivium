from pydantic import BaseModel

from trivium.models.cache import ResponseCache, stable_hash
from trivium.models.embeddings import HashingEmbedder
from trivium.models.llm import MockLLM, extract_json
from trivium.prompts import build_messages, parse_input


class Out(BaseModel):
    answer: str


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure! ```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('prefix {"a": {"b": 2}} suffix') == '{"a": {"b": 2}}'


def test_prompt_roundtrip():
    msgs = build_messages("sys", "do it", {"x": [1, 2], "y": "z"})
    assert msgs[0]["role"] == "system" and "sys" in msgs[0]["content"]
    assert parse_input(msgs) == {"x": [1, 2], "y": "z"}


async def test_mock_llm_caches_by_content(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    llm = MockLLM(cache=cache)
    msgs = build_messages("s", "i", {"question": "What is X?"})
    r1 = await llm.generate(msgs, Out, task="closed_book")
    r2 = await llm.generate(msgs, Out, task="closed_book")
    assert r1.parsed.answer == r2.parsed.answer == "unknown"
    assert not r1.cached and r2.cached and cache.hits == 1
    r3 = await llm.generate(
        msgs, Out, task="closed_book", trial=1
    )  # different trial → separate entry
    assert not r3.cached
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


async def test_hashing_embedder_similarity_ordering():
    emb = HashingEmbedder()
    v = await emb.embed(
        [
            "Which catalyst was used in Experiment 4?",
            "What catalyst was used in Experiment 4?",
            "When was Los Alamos Lab founded?",
        ]
    )
    sim_same = float(v[0] @ v[1])
    sim_diff = float(v[0] @ v[2])
    assert sim_same > 0.6 > sim_diff
    assert abs(float(v[0] @ v[0]) - 1) < 1e-5
