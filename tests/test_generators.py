import pytest

from trivium.facts.store import norm
from trivium.generation import GENERATORS
from trivium.generation.atomic import AtomicGenerator
from trivium.generation.comparison import ComparisonGenerator, comparable_groups
from trivium.generation.derived import DerivedGenerator
from trivium.generation.multihop import MultiHopGenerator
from trivium.generation.true_false import TrueFalseGenerator


@pytest.mark.parametrize(
    "name", [n for n in GENERATORS if n != "persona"]
)  # persona: test_persona.py
async def test_every_strategy_is_available_and_produces_candidates(ctx, name):
    gen = GENERATORS[name](ctx)
    assert gen.available(), name
    cands = await gen.generate(4)
    assert cands, f"{name} produced no candidates"
    for c in cands:
        assert c.strategy == name
        assert c.question.endswith("?") or c.question.startswith("True or False")
        assert c.answer and c.source_fact_ids and c.source_chunk_ids
        assert all(fid in ctx.store for fid in c.source_fact_ids)


async def test_atomic_answer_is_grounded_in_fact(ctx):
    cands = await AtomicGenerator(ctx).generate(6)
    for c in cands:
        f = ctx.store.get(c.source_fact_ids[0])
        assert norm(c.answer) in (norm(f.object.display()), norm(f.subject))


def test_comparable_groups_require_distinct_values(knowledge_base):
    _, _, store = knowledge_base
    groups = comparable_groups(store.all())
    preds = {g[0].predicate for g in groups}
    assert "mean_absolute_error" in preds
    for g in groups:
        assert len({f.object.value for f in g}) >= 2
        assert len({norm(f.subject) for f in g}) == len(g)
    assert comparable_groups(store.all(), cross_document=True)


async def test_comparison_answer_is_deterministic(ctx):
    cands = await ComparisonGenerator(ctx).generate(5)
    assert cands
    for c in cands:
        facts = ctx.store.many(c.source_fact_ids)
        op = c.derivation["op"]
        pick = (min if op == "argmin" else max)(facts, key=lambda f: f.object.value)
        assert norm(c.answer) == norm(pick.subject)
        assert c.reasoning_depth == 2


async def test_derived_difference_is_computed(ctx):
    cands = await DerivedGenerator(ctx).generate(6)
    assert cands
    for c in cands:
        assert c.answer_type == "numeric" and c.derivation["op"] in ("difference", "ratio")
        a, b = ctx.store.many(c.source_fact_ids)
        if c.derivation["op"] == "difference":
            assert c.derivation["value"] == pytest.approx(abs(a.object.value - b.object.value))


async def test_multihop_hides_bridge_and_stores_path(ctx):
    cands = await MultiHopGenerator(ctx).generate(5)
    assert cands
    for c in cands:
        assert c.graph_path and len(c.graph_path) == 5
        bridge = c.graph_path[2]
        assert norm(bridge) not in norm(c.question)
        assert norm(c.answer) == norm(c.graph_path[4])


async def test_true_false_false_statements_are_contradicted(ctx):
    cands = await TrueFalseGenerator(ctx).generate(12)
    answers = {c.answer for c in cands}
    assert answers <= {"True", "False"} and "False" in answers
    for c in cands:
        assert c.answer_type == "boolean" and c.question.startswith("True or False:")
        if c.answer == "False":
            f = ctx.store.get(c.source_fact_ids[0])
            assert norm(c.derivation["asserted_object"]) != norm(f.object.display())
