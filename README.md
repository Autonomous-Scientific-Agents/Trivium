# trivium

Adaptive, grounded question generation for **trivia** and **RAG evaluation**.

`trivium` turns a set of documents (or a topic) into a JSON/JSONL question bank in which
every question has a short, objectively gradable answer, one or more supporting evidence
spans, and a full multi-metric evaluation record. Generation is *adaptive*: candidates are
evaluated on quality, novelty, coverage, document dependence and empirical difficulty, and
the results steer subsequent rounds toward the gaps in the bank.

```
documents/topic → chunks → facts (validated, with provenance) → knowledge graph
      → candidate questions (7 strategies) → multi-metric evaluation → feedback → …
```

The design document is in [`docs/DESIGN.md`](docs/DESIGN.md).

## Install

```bash
uv sync --all-extras          # runtime + openai/anthropic/pypdf extras + dev group
uv run trivium --help
```

Only `pydantic`, `networkx`, `numpy` and `pyyaml` are required; `openai`, `anthropic`,
`pypdf` and `sentence-transformers` are optional extras.

## Quick start

```bash
# from a directory of papers, OpenAI-compatible endpoint
export OPENAI_API_KEY=...            # or OPENAI_BASE_URL=http://localhost:11434/v1 for Ollama/vLLM
uv run trivium generate --input ./papers --num-questions 100 --output questions.json \
    --model gpt-4o-mini --jsonl-dir runs/papers

# from a topic (sources are pulled from Wikipedia, URLs and timestamps are retained)
uv run trivium generate --topic "history of artificial intelligence" -n 50 -o ai_history.json

# offline dry run with the deterministic mock backend (no API calls)
uv run trivium generate --input tests/data --provider mock -n 20 -o /tmp/demo.json --no-cache
```

Optional controls:

```
--easy 30 --medium 40 --hard 30           difficulty quotas (sum implies --num-questions)
--question-types short_answer,true_false,multiple_choice
--max-answer-words 8
--require-document-dependence             strict RAG mode: reject closed-book-answerable questions
--multi-hop-ratio 0.25 --cross-document-ratio 0.10
--mode rag|trivia                         composite-score weighting
--seed 1234 --batch-size 8 --max-rounds 60 --max-candidates 500
--config trivium.yaml                     per-role models, thresholds, chunking (examples/config.yaml)
```

If `--num-questions` is omitted the target is estimated from the number of validated facts,
relational facts and two-hop graph paths (design §4), capped by `generation.max_questions`.

Other commands: `trivium extract -i docs/ -o facts.jsonl` (ingestion + facts only) and
`trivium stats questions.json` (dataset-level metrics).

## Model roles

Every LLM role is independently configurable (design §29). Roles not listed fall back to
`models.default`; `difficulty_small` / `difficulty_medium` are optional — without them
difficulty is a proxy from closed-book accuracy and reasoning depth.

```yaml
models:
  default: {provider: openai, model: gpt-4o, temperature: 0}
  roles:
    fact_extractor:    {model: gpt-4o}
    closed_book:       {model: gpt-4o}
    difficulty_small:  {model: gpt-4o-mini}
    difficulty_medium: {model: gpt-4o}
    judge:             {model: gpt-4o-mini}
  embeddings: {provider: openai, model: text-embedding-3-small}   # or hashing (no deps)
```

Providers: `openai` (any OpenAI-compatible server: OpenAI, vLLM, Ollama, ALCF inference
endpoints via `base_url`), `anthropic`, and `mock` (deterministic, offline).
Environment overrides: `TRIVIUM_PROVIDER`, `TRIVIUM_MODEL`, `TRIVIUM_BASE_URL`.

## What gets evaluated

| stage | metrics | hard constraint (default) |
|---|---|---|
| structural | answer length, answer-in-question, embedding + fact-overlap redundancy | duplicate ≥ 0.90 |
| quality rubric | answerability, evidence entailment, ambiguity, conciseness, clarity, leakage, specificity, interestingness | entailment ≥ 0.9, answerability ≥ 0.9, ambiguity ≤ 0.2 |
| open book | strong model + gold evidence | accuracy ≥ 0.8 |
| closed book | strong model, no evidence, N trials | (strict RAG mode only) |
| document dependence | open − closed | (strict RAG mode only) |
| difficulty | small / medium / large tier accuracies | thresholds configurable |
| retrieval difficulty | rank of gold chunk under embedding retrieval | informational |
| composite | mode-weighted quality score | ≥ 0.75 |

Rejected candidates are kept (`rejected.jsonl` in the JSONL bundle) with their reasons;
rejection statistics feed strategy weights and the next round's generation prompt.

## Generation strategies

`atomic`, `relationship`, `comparison`, `derived` (difference/ratio with recorded
dependencies), `multi_hop` (two-edge graph paths, path stored), `cross_document`,
`true_false` (perturbed facts verified as *contradicted*, not merely absent) and a
`multiple_choice` converter with per-distractor validation.

## Library use

```python
import asyncio
from trivium import Config, generate_dataset

cfg = Config.load("trivium.yaml", inputs=["./papers"], generation={"num_questions": 50})
dataset, run = asyncio.run(generate_dataset(cfg))
print(dataset.dataset_metrics)
```

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests
```

The test-suite runs entirely on the mock backend.

## License

Apache-2.0.
