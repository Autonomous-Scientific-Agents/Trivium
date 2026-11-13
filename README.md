# trivium

Adaptive, grounded question generation for **trivia** and **RAG evaluation**.

`trivium` turns a set of documents (or a topic) into a JSON/JSONL question bank in which
every question has a short, objectively gradable answer, one or more supporting evidence
spans, and a full multi-metric evaluation record. Generation is *adaptive*: candidates are
evaluated on quality, novelty, coverage, document dependence and empirical difficulty, and
the results steer subsequent rounds toward the gaps in the bank.

```
documents/topic → chunks → facts (validated, with provenance) → knowledge graph
      → candidate questions (8 strategies) → multi-metric evaluation → feedback → …
```

The design document is in [`docs/DESIGN.md`](docs/DESIGN.md).

## Install

```bash
uv sync --all-extras          # runtime, every extra and the dev group
uv run trivium --help
```

Only `pydantic`, `networkx`, `numpy`, `pyyaml` and `httpx` are required. Extras: `openai`
(also needed for the `asksage` and `alcf` providers), `anthropic`, `pdf` (pypdf), `embeddings`
(sentence-transformers), `web` (the review UI) and `all` (openai, anthropic and pdf).

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
--mode rag|trivia                         what is generated and how it is scored (see Trivia mode)
--best K --select-by trivia               keep the K best of a larger pool (see Best-of selection)
--prefer-difficulty easy|medium|hard      steer generation toward a difficulty from round one
--persona "a new user ..."                who asks; repeatable (see Personas)
--seed 1234 --batch-size 8 --max-rounds 60 --max-candidates 500
--provider alcf --model ... --base-url ... default model for every role
--config trivium.yaml                     per-role models, thresholds, chunking (examples/config.yaml)
--max-chunks 20 --no-validate-facts       debugging: fewer chunks, skip fact validation
```

If `--num-questions` is omitted the target is estimated from the number of validated facts,
relational facts and two-hop graph paths (design §4), capped by `generation.max_questions`.

Other commands: `trivium extract -i docs/ -o facts.jsonl` (ingestion + facts only),
`trivium stats questions.json` (dataset-level metrics), `trivium serve` (the review UI) and
`trivium feedback-report` (summarises review ratings); see Review UI.

If the provider refuses service (Argo's usage notice, Ask Sage's quota or revoked-key errors),
the run stops at once without retrying. Questions accepted before that are still written; if it
happens before any question is accepted, `generate` exits with code 3.

## Model roles

Every LLM role is independently configurable (design §29): `fact_extractor`, `fact_validator`,
`trivia_scorer`, `question_generator`, `quality_evaluator`, `judge`, `closed_book`, `open_book`,
`difficulty_small`, `difficulty_medium` and `difficulty_large`. Roles not listed fall back to
`models.default`, except `difficulty_small` / `difficulty_medium`, which are off unless set;
without them difficulty is a proxy from closed-book accuracy and reasoning depth.

```yaml
models:
  default: {provider: openai, model: gpt-4o, temperature: 0}
  roles:
    fact_extractor:    {model: gpt-4o}
    closed_book:       {model: gpt-4o}
    difficulty_small:  {model: gpt-4o-mini}
    difficulty_medium: {model: gpt-4o}
    judge:             {model: gpt-4o-mini}
  embeddings: {provider: openai, model: text-embedding-3-small}
```

Embedding providers: `hashing` (the default; no dependencies, lexical only),
`sentence-transformers` (local, needs the `embeddings` extra), `openai`, `argo`, `asksage` and
`alcf`. Embeddings drive duplicate detection, retrieval difficulty and the persona strategy's
passage search, so use a real embedding model for anything beyond a smoke test.

Providers: `openai` (any OpenAI-compatible server: OpenAI, vLLM, Ollama, ALCF inference
endpoints via `base_url`), `anthropic`, `argo` (Argonne's Argo Gateway, see below), `asksage` (Ask Sage, see below),
`alcf` (open-weight models on the ALCF inference endpoints, see below), and `mock`
(deterministic, offline). Environment overrides: `TRIVIUM_PROVIDER`, `TRIVIUM_MODEL`, `TRIVIUM_BASE_URL`.

**Mixing providers.** Each role can use a different provider, e.g. open-weight models on ALCF to
write questions and a different model family on Ask Sage or Argo to grade them
(`examples/config.mixed.yaml`). Role entries are merged onto `default`. A role that switches
provider keeps `default`'s temperature, `max_tokens` and concurrency, but not its `base_url`,
`api_key_env` or `options`. Embeddings are configured separately under `models.embeddings`.

**Avoid self-evaluation.** By default the question generator, quality evaluator, open-book
answerer and large difficulty tier all resolve to `models.default`, so a model grades its own
questions. For datasets you intend to publish, set `roles.quality_evaluator`, `roles.judge` and
`roles.open_book` to a model from a different family than `question_generator`
(e.g. generate with `gpt41` and evaluate with `claudesonnet45` on Argo, or generate on ALCF and
evaluate on Ask Sage as in `examples/config.mixed.yaml`).

## Best-of selection: "the single hardest, most interesting question"

Acceptance is a threshold test, so `-n 1` returns the *first* question that clears the bar, not
the best one. `--best K` instead generates an accepted pool (default `8*K`, at least 10; or `-n`),
ranks it and keeps only the top K, moving the rest to `rejected.jsonl` as `not_selected`:

```bash
uv run trivium generate --input paper.pdf --best 1 --mode trivia -o best.json
```

`--best` also steers the adaptive loop toward hard questions from the first round (multi-hop,
derived and comparison strategies are up-weighted and the generator is told to avoid
single-lookup questions). The ranking criterion (`--select-by`) defaults to `trivia` in
`--mode trivia` and `challenge` otherwise:

| criterion | score |
|---|---|
| `challenge` | 0.4 · difficulty label (extreme > hard > medium > easy) + 0.3 · failure rate of the small/medium tiers + 0.2 · closed-book failure + 0.1 · reasoning depth |
| `interest` | 0.7 · rubric interestingness + 0.15 · specificity + 0.15 · clarity |
| `trivia` | 0.55 · challenge + 0.45 · interest |
| `quality` | the composite quality score |

Every question carries `evaluation.selection_score` and `selection_rank`, so a different
criterion can be applied offline to the saved pool. Configure `difficulty_small` /
`difficulty_medium` roles for a real challenge signal; without tiers the closed-book result is
the only proxy.

## Trivia mode

`--mode trivia` changes what is generated, not only how it is weighted:

| stage | what happens |
|---|---|
| fact scoring | a brief per document (summary + notable nuggets from the head and tail of the text) and a `trivia_score` for every fact; sampling is weighted by `(0.1 + score)²`, so a key result is ~10× as likely to be asked about as a run label. Role `trivia_scorer`. |
| writing | the question writer also sees the surrounding passage and the brief, and must declare an `answer_kind` (person, place, organization, name, term, number, date, title). |
| answer form | default `max_answer_words` 4 (8 in RAG mode); rejects `answer_kind_other`, `generic_answer` ("larger matrix", "either method") and `clause_answer` ("cases where …", "only at the beginning of …"). Rejects `answer_not_in_evidence` when the answer is not in the cited facts, because the writer also sees the surrounding passage and can take the answer from it. |
| standalone | rejects `local_label` ("Run 2", "the larger matrix", "Experiment 4") and rubric `standalone` below 0.6. `source_reference` ("Table 3", "the authors", document ids) and `placeholder_answer` (a template slot copied from a command example, such as `<fs1:fs2>`, `/path/to/…` or `YOUR_PROJECT`) are rejected in every mode and for every question type. |
| multiple choice | distractors from typed pools: `in_document` (same predicate, same unit, close magnitude, or the competing subjects), `misconception` (wrong closed-book answers), `near_miss` (same unit and precision); each audited for type, plausibility and correctness. Items the source lists together with the answer are never used as wrong options: when a script installs "pip, wheel, setuptools" alongside numpy, `wheel` would also be a correct answer. Option sets are rejected when the answer `stands out` in length or form, or when an options-only probe (no question shown) finds it twice. In trivia mode an MC item whose chance-corrected closed-book accuracy exceeds 0.67 falls back to short answer. Distractor kinds and per-option closed-book picks are recorded. |
| true/false | false statements by `subject_swap`, `qualifier_swap`, `comparison_flip`, `llm_edit` (one targeted edit), numeric perturbation or sibling swap; true statements are paraphrased, false ones stay close to the source wording; at most 25 words. Every statement is entailment-checked. |
| `--best K` | interestingness comes from a Swiss pairwise tournament ("which is the better trivia question?", A/B order randomised) instead of absolute rubric scores, which cluster around 0.7. |

Fallback reasons for multiple-choice conversion are reported in `strategy_stats.multiple_choice_fallbacks`.

Configuration values, such as environment variables, command-line flags and "X is set to 12" facts, are never compared or used in arithmetic in any mode. A question like "NRANKS_PER_NODE divided by NTHREADS" is filler.

## Personas

```bash
trivium generate -i docs/ --mode trivia --best 6 \
  --persona "a new user who just got their first allocation and has never used PBS"
```

A persona describes who is asking. You can pass `--persona` more than once, and the review UI's new-run form has a persona box too. A persona changes four things:

- **Fact scoring:** facts are rated for what this audience needs to know, not only for how remarkable they are.
- **Question writing:** every question writer is told to ask what the persona would ask, in their words.
- **Ranking:** the `--best` tournament prefers the question the audience would find more useful.
- **Questions from gaps:** a new `persona` strategy asks questions without seeing the text.
  1. The persona sees only an outline of the sources: titles, section headings and a summary.
  2. It asks questions in five styles in turn: factual, procedural, troubleshooting, limits and comparative.
  3. Each question is matched to the most similar passages. A model pulls out a short answer and a verbatim quote, and sharpens the question so it has exactly one answer.
  4. The question is kept only if the quote appears in the passage and contains the answer.

The `persona` strategy's share starts at `generation.persona_weight` (default 0.5, against 0.25 for `atomic`) and then adapts to its acceptance rate like every strategy. Raise it when the persona questions are the ones you want; in the Aurora runs they were the most useful.

Questions that the sources answer only partly, or not at all, are reported in `strategy_stats.persona_gaps` (up to 200) and on the run page. They show what the documentation does not cover.

`evaluation.lexical_overlap` is the share of a question's content words that appear in its evidence, with its average in `dataset_metrics`. Questions asked from gaps should score lower than ones rephrased from the text.

## Review UI

```bash
uv sync --extra web            # or: pip install 'trivium[web]'
trivium serve --config my.yaml # http://127.0.0.1:8765/, or the next free port
```

The browser UI lets you:

- **upload sources** (drag and drop files, or paste text) and start a run, with mode, number of questions, best-of K, question types, provider, model, seed and personas as options;
- **retry** a finished or failed run with the same settings. Earlier ratings are archived to `feedback.<timestamp>.json`, because the new questions reuse the old ids;
- **follow the run** while it generates, through a live log;
- **review the questions**: each card shows the options (with the correct one marked), the answer, accepted alternatives, the evidence, the evaluation scores and how distractors or falsifications were made;
- **rate each question** with a thumb up or down and a free-form comment. Feedback saves automatically.

Filters show unrated, 👍, 👎 or commented questions, or one question type. Quiz mode hides answers until clicked. Keyboard shortcuts: `j`/`k` move between questions, `u`/`d` rate, `c` jumps to the comment box, `a` reveals the answer.

Runs live under `--workspace` (default `runs/`), one folder per run:

- `sources/` holds the uploaded files;
- `questions.json` and `bundle/` hold the generated dataset;
- `generate.log` is the run's log;
- `job.json` records the run's settings and status;
- `feedback.json` stores the ratings and comments.

Run folders written by `trivium generate` are listed too, so you can review earlier runs. **Export feedback** downloads one JSONL line per question with the question, answer, rating, comment and reviewer.

`trivium feedback-report [run ...]` summarises your ratings across runs:

- the share of questions rated 👍, broken down by run, type, strategy and persona question style;
- how well each automatic score agrees with your ratings. Agreement is measured as AUC: the probability that a random 👍 question scores higher than a random 👎 one, where 0.5 means no better than chance;
- every 👎 comment.

Use it to check whether a change helps, and which automatic scores are worth ranking by.

Runs execute one at a time, in the order they were started. `--config` supplies every setting the form does not set, such as provider credentials and thresholds. There is no authentication, so keep the default `--host 127.0.0.1` unless something in front of the UI handles authentication.

### Argo Gateway (Argonne)

The `argo` provider talks to the native Argo endpoints (chat + embeddings), so it works against
production and with every model Argo hosts (`gpt4o`, `gpt41`, `gpt41mini`, `gpt41nano`, `gpt5*`,
`gpto3`, `gpto4mini`, `claudeopus46`, `claudesonnet45`, `claudehaiku45`, `gemini25pro`,
`gemini25flash`, …; embeddings `v3small`, `v3large`, `ada002`). Reasoning models automatically get
`max_completion_tokens` instead of sampling parameters. You must be on the Argonne network/VPN
and pass your ANL domain username:

```bash
export ARGO_USER=<your ANL domain username>
uv run trivium generate --input tests/data --config examples/config.argo.yaml -n 15 \
    --jsonl-dir runs/argo -o runs/argo/questions.json -v
```

`examples/config.argo.yaml` uses `gpt41` by default with `gpt41nano`/`gpt41mini` as the small and
medium difficulty tiers and `v3small` embeddings. Per-role `options: {env: dev}` switches a role to
the dev gateway (e.g. for models only available there). Argo's OpenAI-compatible shim
(`https://apps-dev.inside.anl.gov/argoapi/v1`, `api_key` = ANL username) also works via the
`openai` provider with `api_key_env: ARGO_USER`, but it is dev-only and has no embeddings.

### Ask Sage

The `asksage` provider (chat and embeddings) uses Ask Sage's OpenAI-compatible API with your Ask
Sage API key as a Bearer token. It defaults to ANL's instance
(`https://api.asksage.anl.gov/server/openai/v1`) and reads the key from `$ASK_SAGE_API` (or
`$ASK_SAGE_API_KEY`, `$ASKSAGE_API_KEY`, or the role's `api_key_env`). It never falls back to
`$OPENAI_API_KEY`. For another instance, set `base_url` or `$ASK_SAGE_BASE_URL`.

```bash
export ASK_SAGE_API=<your Ask Sage API key>
uv run trivium generate --input tests/data --config examples/config.asksage.yaml -n 12 \
    --jsonl-dir runs/sage -o runs/sage/questions.json
uv run trivium serve --config examples/config.asksage.yaml
```

The model list (`GET /models`) shows more models than a given key may use. A model your key
cannot use fails at once with "Requested model is not available". `examples/config.asksage.yaml`
uses models that answered in November 2026: `gpt-6-luna` by default, `gpt-5.6-terra` as the judge,
`gpt-5.4-nano`/`gpt-5.6-luna` as the small and medium difficulty tiers, and
`text-embedding-3-large` embeddings (`text-embedding-3-small` is unavailable on the ANL instance).
Ask Sage's "Token quota exceeded" and "API key has been revoked or expired" errors stop the run
without retrying, like Argo's usage notice (see Quick start).

### ALCF inference endpoints

The `alcf` provider (chat and embeddings) uses the open-weight models served on ALCF's
[inference endpoints](https://docs.alcf.anl.gov/services/inference-endpoints/). Pick the cluster
with `options: {cluster: ...}`: `sophia` (vLLM, the default), `metis`, `minerva` or `tara`. For
models that take a reasoning level, `options: {reasoning_effort: low}` saves tokens. The access
token comes from `alcf-tokens get-token inference` (or `$ALCF_ACCESS_TOKEN`). Tokens last 48 hours
and are refreshed automatically, including during a run.

```bash
uv tool install alcf-tokens && alcf-tokens login inference   # once; ALCF asks for a re-login every 30 days
uv run trivium generate --input tests/data --config examples/config.alcf.yaml -n 12 \
    --jsonl-dir runs/alcf -o runs/alcf/questions.json
```

`examples/config.alcf.yaml` generates with `openai/gpt-oss-120b` on Sophia and grades with
`mistral-large-4-nvfp4` on Minerva, a different model family; the cluster is set per role with
`options: {cluster: minerva}`. It uses Llama 3.1 8B, `gpt-oss-20b`
and `gpt-oss-120b` as the difficulty tiers and `Salesforce/SFR-Embedding-Mistral` embeddings (avoid `google/embeddinggemma-300m`: on this endpoint it rates unrelated texts about 0.96 similar, so every question looks like a duplicate). All of
these answered at once in November 2026. Models that are not "always hot" are loaded on the first
request and can take many minutes to answer. The service allows 20 requests per second per caller.

## What gets evaluated

| stage | metrics | hard constraint (default) |
|---|---|---|
| structural | answer length, answer-in-question, source references, placeholder answers, generic answers, embedding + fact-overlap redundancy (trivia: local labels, clause answers, answer kind) | duplicate ≥ 0.90 |
| quality rubric | answerability, evidence entailment, ambiguity, conciseness, clarity, leakage, specificity, standalone, interestingness | entailment ≥ 0.9, answerability ≥ 0.9, ambiguity ≤ 0.2 (trivia: standalone ≥ 0.6) |
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
`true_false` (minimal edits of facts verified as *contradicted*, not merely absent), `persona`
(questions asked from the outline, then grounded; needs `--persona`, see Personas) and a
`multiple_choice` converter with typed, per-distractor-validated distractors (see Trivia mode).

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
