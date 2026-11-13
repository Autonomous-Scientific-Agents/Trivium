"""Command-line interface: ``trivium generate | extract | stats | serve``."""

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
    p.add_argument(
        "--provider", help="default LLM provider: openai | anthropic | argo | asksage | alcf | mock"
    )
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
    src = g.add_mutually_exclusive_group()  # may also come from the YAML config (inputs:/topic:)
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
    g.add_argument(
        "--best",
        type=int,
        metavar="K",
        help="generate a pool (default 8*K, min 10; or -n), rank it and keep only the K best; "
        "steers generation toward hard questions",
    )
    g.add_argument(
        "--select-by",
        choices=["challenge", "interest", "trivia", "quality"],
        help="ranking used by --best (default: trivia in --mode trivia, else challenge)",
    )
    g.add_argument(
        "--prefer-difficulty",
        choices=["easy", "medium", "hard"],
        help="steer the adaptive loop toward this difficulty from the first round",
    )
    g.add_argument("--easy", type=int)
    g.add_argument("--medium", type=int)
    g.add_argument("--hard", type=int)
    g.add_argument("--question-types", help="comma list: short_answer,true_false,multiple_choice")
    g.add_argument("--max-answer-words", type=int)
    g.add_argument(
        "--persona",
        action="append",
        metavar="TEXT",
        help="who asks (repeatable), e.g. 'a new user on their first allocation'; steers scoring, "
        "writing and ranking, and adds gap-driven questions asked from the outline only",
    )
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

    fr = sub.add_parser(
        "feedback-report",
        help="summarise review-UI ratings: 👍 rate by strategy/type and agreement of scores",
    )
    fr.add_argument("runs", nargs="*", help="run ids (default: every run in the workspace)")
    fr.add_argument("--workspace", default="runs")

    w = sub.add_parser(
        "serve",
        help="web UI: upload sources, generate, review questions and leave feedback",
        description="Start the review web UI (needs: pip install 'trivium[web]'). Runs and "
        "feedback are stored under --workspace; existing run folders there are listed too.",
    )
    w.add_argument("--workspace", default="runs", help="directory holding runs (default: runs)")
    w.add_argument("--config", help="YAML config used for every run started from the UI")
    w.add_argument("--provider", help="default LLM provider shown in the UI")
    w.add_argument("--model", help="default model shown in the UI")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument(
        "--port",
        type=int,
        help="default: the first free port from 8765 (fails if a given port is busy)",
    )
    w.add_argument("-v", "--verbose", action="store_true")
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
    if getattr(args, "best", None) is not None:
        gen["select_top"] = args.best
    if getattr(args, "select_by", None):
        gen["select_by"] = args.select_by
    if getattr(args, "prefer_difficulty", None):
        gen["prefer_difficulty"] = args.prefer_difficulty
    if getattr(args, "require_document_dependence", False):
        gen["require_document_dependence"] = True
    if getattr(args, "persona", None):
        gen["personas"] = [p.strip() for p in args.persona if p.strip()]
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
    if not cfg.inputs and not cfg.topic:
        raise SystemExit(
            "generate: provide --input/--topic or set inputs:/topic: in the config file"
        )
    from trivium.models.llm import ProviderAccessError

    try:
        dataset, run = asyncio.run(generate_dataset(cfg))
    except ProviderAccessError as exc:
        print(f"generate: stopped, the model provider refused service:\n{exc}", file=sys.stderr)
        return 3
    if run.stop_reason == "provider_refused":
        print(
            f"generate: stopped early, the model provider refused service:\n{run.error}\n"
            f"wrote the {len(dataset.questions)} question(s) accepted before that",
            file=sys.stderr,
        )
    m = dataset.dataset_metrics
    print(
        json.dumps(
            {
                "output": cfg.output,
                "accepted": len(dataset.questions),
                **(
                    {"selected_by": run.selected_by, "pool": run.target}
                    if cfg.generation.select_top
                    else {}
                ),
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
    if not cfg.inputs and not cfg.topic:
        raise SystemExit(
            "extract: provide --input/--topic or set inputs:/topic: in the config file"
        )

    async def _run() -> int:
        models = ModelRegistry(cfg)
        try:
            docs = await load_documents(cfg)
            chunks, store = await build_knowledge_base(cfg, models, docs)
        finally:
            await models.aclose()
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


def _port_free(host: str, port: int) -> bool:
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(host: str, port: int | None, start: int = 8765, tries: int = 20) -> int:
    """The requested port, or the first free one from ``start`` when none was requested."""
    if port is not None:
        if not _port_free(host, port):
            raise SystemExit(f"serve: port {port} on {host} is already in use; pick another --port")
        return port
    for p in range(start, start + tries):
        if _port_free(host, p):
            return p
    raise SystemExit(f"serve: no free port in {start}-{start + tries - 1}; pass --port")


def cmd_feedback_report(args: argparse.Namespace) -> int:
    from trivium.output.feedback import collect, report

    rows = collect(args.workspace, args.runs or None)
    print(json.dumps(report(rows), indent=2, ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from trivium.web.app import create_app
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"serve: {exc}. Install the web extra: pip install 'trivium[web]'"
        ) from None
    args.port = pick_port(args.host, args.port)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logging.getLogger("trivium").warning(
            "the review UI has no authentication; anyone who can reach %s:%d can upload files "
            "and start LLM jobs",
            args.host,
            args.port,
        )
    defaults = {k: v for k, v in (("provider", args.provider), ("model", args.model)) if v}
    app = create_app(args.workspace, base_config=args.config, defaults=defaults)
    print(f"trivium review UI: http://{args.host}:{args.port}/  (workspace: {args.workspace})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    commands = {
        "generate": cmd_generate,
        "extract": cmd_extract,
        "stats": cmd_stats,
        "serve": cmd_serve,
        "feedback-report": cmd_feedback_report,
    }
    return commands[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
