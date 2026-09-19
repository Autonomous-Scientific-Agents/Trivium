"""Command-line interface: ``trivium generate``, ``trivium extract``, ``trivium stats``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from trivium import __version__
from trivium.config import Config


def _common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config file (see examples/config.yaml)")
    p.add_argument("--provider", help="default LLM provider: openai | anthropic | mock")
    p.add_argument("--model", help="default model name for all roles")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint URL")
    p.add_argument("--no-cache", action="store_true", help="disable the on-disk response cache")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="trivium",
        description="Adaptive grounded question generation for trivia and RAG benchmarks.",
    )
    ap.add_argument("--version", action="version", version=f"trivium {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a question dataset from documents or a topic")
    src = g.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", nargs="+", help="files or directories (.txt .md .pdf .html)")
    src.add_argument("--topic", help="topic string; sources are acquired from Wikipedia")
    g.add_argument(
        "--num-questions", "-n", type=int, help="target number of questions (auto if omitted)"
    )
    g.add_argument("--output", "-o", default="questions.json")
    g.add_argument(
        "--jsonl-dir",
        help="also write a JSONL bundle (metadata, questions, facts, documents, rejected)",
    )
    g.add_argument("--mode", choices=["rag", "trivia"], default=None)
    g.add_argument("--easy", type=int)
    g.add_argument("--medium", type=int)
    g.add_argument("--hard", type=int)
    g.add_argument("--question-types", help="comma list: short_answer,true_false,multiple_choice")
    g.add_argument("--max-answer-words", type=int)
    g.add_argument("--require-document-dependence", action="store_true")
    g.add_argument("--multi-hop-ratio", type=float)
    g.add_argument("--cross-document-ratio", type=float)
    g.add_argument("--seed", type=int)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--max-rounds", type=int)
    g.add_argument("--max-candidates", type=int, help="candidate budget (cost limit)")
    g.add_argument("--max-chunks", type=int, help="debug: only use the first N chunks")
    g.add_argument("--no-validate-facts", action="store_true")
    _common_model_args(g)

    e = sub.add_parser("extract", help="only run ingestion + fact extraction; write facts.jsonl")
    e.add_argument("--input", "-i", nargs="+", required=True)
    e.add_argument("--output", "-o", default="facts.jsonl")
    e.add_argument("--max-chunks", type=int)
    e.add_argument("--no-validate-facts", action="store_true")
    _common_model_args(e)

    s = sub.add_parser("stats", help="print dataset-level metrics of a questions.json file")
    s.add_argument("path")
    return ap


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    gen: dict[str, Any] = {}
    for key in (
        "num_questions",
        "easy",
        "medium",
        "hard",
        "max_answer_words",
        "multi_hop_ratio",
        "cross_document_ratio",
        "seed",
        "batch_size",
        "max_rounds",
        "max_candidates",
    ):
        if getattr(args, key, None) is not None:
            gen[key] = getattr(args, key)
    if getattr(args, "require_document_dependence", False):
        gen["require_document_dependence"] = True
    if getattr(args, "question_types", None):
        gen["question_types"] = [t.strip() for t in args.question_types.split(",") if t.strip()]
    models: dict[str, Any] = {"default": {}}
    if args.provider:
        models["default"]["provider"] = args.provider
    if args.model:
        models["default"]["model"] = args.model
    if args.base_url:
        models["default"]["base_url"] = args.base_url
    if args.provider == "mock":
        models.setdefault("embeddings", {})["provider"] = "hashing"
    extraction: dict[str, Any] = {}
    if getattr(args, "max_chunks", None):
        extraction["max_chunks"] = args.max_chunks
    if getattr(args, "no_validate_facts", False):
        extraction["validate_facts"] = False
    cache: dict[str, Any] = {}
    if args.no_cache:
        cache["enabled"] = False
    if args.cache_dir:
        cache["dir"] = args.cache_dir
    ov: dict[str, Any] = {
        "generation": gen,
        "models": models,
        "extraction": extraction,
        "cache": cache,
    }
    if getattr(args, "input", None):
        ov["inputs"] = args.input
    if getattr(args, "topic", None):
        ov["topic"] = args.topic
    if getattr(args, "output", None):
        ov["output"] = args.output
    if getattr(args, "jsonl_dir", None):
        ov["jsonl_dir"] = args.jsonl_dir
    if getattr(args, "mode", None):
        ov["mode"] = args.mode
    if args.verbose:
        ov["log_level"] = "DEBUG"
    return ov


def cmd_generate(args: argparse.Namespace) -> int:
    from trivium.orchestration.controller import generate_dataset

    cfg = Config.load(args.config, **_overrides(args))
    dataset, run = asyncio.run(generate_dataset(cfg))
    m = dataset.dataset_metrics
    print(
        json.dumps(
            {
                "output": cfg.output,
                "accepted": len(dataset.questions),
                "candidates": run.n_candidates,
                "rounds": run.rounds,
                "stop_reason": run.stop_reason,
                "difficulty": m.get("difficulty_distribution"),
                "types": m.get("type_distribution"),
                "avg_quality": m.get("average_quality"),
                "avg_document_dependence": m.get("average_document_dependence"),
                "llm_calls": run.llm_calls,
            },
            indent=2,
        )
    )
    return 0 if dataset.questions else 1


def cmd_extract(args: argparse.Namespace) -> int:
    from trivium.models.registry import ModelRegistry
    from trivium.orchestration.controller import build_knowledge_base, load_documents

    cfg = Config.load(args.config, **_overrides(args))

    async def _run() -> int:
        models = ModelRegistry(cfg)
        try:
            docs = await load_documents(cfg)
            chunks, store = await build_knowledge_base(cfg, models, docs)
        finally:
            models.close()
        store.save_jsonl(args.output)
        print(
            json.dumps(
                {
                    "documents": len(docs),
                    "chunks": len(chunks),
                    "facts": len(store),
                    "output": args.output,
                },
                indent=2,
            )
        )
        return 0

    return asyncio.run(_run())


def cmd_stats(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "dataset": {k: v for k, v in data["dataset"].items() if k != "config"},
                "dataset_metrics": data.get("dataset_metrics"),
                "coverage": data.get("coverage"),
                "strategy_stats": data.get("strategy_stats"),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return {"generate": cmd_generate, "extract": cmd_extract, "stats": cmd_stats}[args.command](
        args
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
