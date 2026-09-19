# Code review: trivium

Date: 2026-09-19
Scope: all of `src/trivium` (about 5,000 lines) at commit `b440094` plus the uncommitted Argo changes (`models/argo.py`, `tests/test_argo.py`, `examples/config.argo.yaml` and related edits).
State at review time: 49 tests pass, `ruff check` passes. No code was changed as part of this review.

The pipeline from ingest through generation, evaluation and the adaptive loop is well organised, and mock mode keeps it testable offline. There are three confirmed bugs that corrupt gold answers or grading; they should be fixed before generating datasets for real use.

## 1. Bugs reproduced

### 1.1 Multiple-choice answer "A" is always graded wrong

Location: `src/trivium/evaluation/grading.py:15` (`_ARTICLES`), used by `normalize_answer` and `match_answer`.

- `_ARTICLES` strips the word "a", so `normalize_answer("A")` returns an empty string.
- `match_answer("A", ["A", "Paris"], "choice")` returns `False`, while "B" works.
- Any multiple-choice item whose shuffled correct option lands on A fails the open-book check and silently falls back to short-answer (`src/trivium/orchestration/controller.py:304`).
- The output dataset therefore almost never has A as the correct letter, and the tier accuracies for those items are wrong.
- `tests/test_grading.py:19` only tests "B". The mock pipeline test hides the problem because the multiple-choice fallback swallows the failure.

Fix: handle `choice` before article stripping, or skip stripping for single-token answers.

### 1.2 `_fmt` mangles derived answers of 1000 or more that end in zeros

Location: `src/trivium/generation/derived.py:23`.

- With `digits == 0`, `rstrip("0")` removes integer zeros.
- `_fmt(1000)` returns `"1"`, `_fmt(1200)` returns `"12"`, `_fmt(250000)` returns `"25"`.
- That string becomes the gold `answer` of the derived question.

Fix: only strip trailing zeros when the string contains a ".".

### 1.3 `answers_match` substring logic lets the LLM override a fixed answer

Location: `src/trivium/generation/base.py:269`, used at `src/trivium/generation/base.py:257`.

- `answers_match("2.5", "12.5")`, `("1", "10")` and `("no", "know")` all return `True` because the comparison is a string substring test.
- In `ask_llm`, a "match" keeps the LLM's answer instead of the deterministically computed one, so a wrong number can replace the derived value.
- The same function also does the grounding check in `generation/atomic.py` and the distractor filtering in `generation/multiple_choice.py`.

Fix: always set `gq.answer = fixed_answer` when a fixed answer is given, and make the match token-based rather than substring-based.

## 2. Likely issues

### 2.1 Grading is too permissive

Location: `src/trivium/evaluation/grading.py:69-83`.

Reproduced results:

| Prediction | Reference | Type | Result |
|---|---|---|---|
| `Table 4` | `Figure 4` | short_answer | match |
| `12 miles` | `12 km` | numeric | match |
| `Paris or London` | `Paris` | short_answer | match |
| `not Paris` | `Paris` | short_answer | match |

In short-answer mode any shared first number matches, units are ignored, and the containment rule accepts hedges and negations. This inflates the accuracies that drive difficulty labels and document dependence.

Suggested fix: apply the numeric path only when the reference is essentially a number, compare units, and reject predictions containing "or" / "not" around the reference.

### 2.2 Retry backoff sleeps while holding the semaphore

Location: `src/trivium/models/llm.py:133-147`.

The `except` block (with `asyncio.sleep`) is inside `async with self._sem`, so a failing call blocks a concurrency slot for up to 8 seconds. Non-retryable errors (auth failures, HTTP 400s, `ArgoConfigError`) are also retried three times.

### 2.3 Infrastructure failures look like model failures

- `src/trivium/evaluation/answering.py:60` returns `""` on `LLMError`. A closed-book outage then makes questions look document-dependent, and an open-book outage rejects them as `large_model_failed_open_book`.
- `OpenAIEmbedder` and `ArgoEmbedder` have no retry. An exception in `Evaluator.evaluate` escapes `asyncio.gather` in the controller and aborts the run with nothing written.
- A re-run replays chat responses from the cache. Embeddings replay only with the OpenAI or Argo embedders; the default hashing embedder recomputes locally. All of this depends on the cache being enabled.

### 2.4 Same-round duplicate re-check is weaker than the main one

Location: `src/trivium/orchestration/controller.py:222`.

The sequential re-check only tests `sim >= duplicate_similarity`, with no judge and no fact-overlap rule. Two questions from the same fact in one batch can both be accepted.

### 2.5 CLI and config precedence

- `generate` requires `--input` or `--topic` even when the YAML config already has `inputs:` (`src/trivium/cli.py:36`).
- `TRIVIUM_PROVIDER` / `TRIVIUM_MODEL` / `TRIVIUM_BASE_URL` override explicit CLI flags (`src/trivium/config.py:235`), which is an unusual precedence.
- The `--provider` help text does not mention `argo`.

### 2.6 OpenAI provider and reasoning models

Location: `src/trivium/models/llm.py:223`.

`OpenAILLM` sends `max_tokens` and `temperature` to every model. Argo handles reasoning models (`max_completion_tokens`, no sampling parameters), but plain OpenAI `o*` / `gpt-5` models will fail.

### 2.7 Cache and registry keys omit `max_tokens`

- `ModelSpec.key()` (`src/trivium/config.py:48`) omits `max_tokens`, `max_concurrency` and `api_key_env`, so roles that differ only in those share one instance.
- The LLM cache key (`src/trivium/models/llm.py:115`) omits `max_tokens`, so a truncated response is reused after the limit is raised.

### 2.8 Username leaks into the output dataset

Location: `src/trivium/orchestration/controller.py:76`.

The output JSON embeds the full config, including `models.*.options.user` (the ANL username). Worth excluding if datasets are shared.

### 2.9 Chunk offsets

Location: `src/trivium/ingestion/chunker.py:55`.

- Pieces of a split paragraph are re-joined with spaces, so `find` fails when the original had line breaks and every piece gets the paragraph start offset.
- A single sentence longer than `max_chars` is never split.

## 3. Minor and cleanup

- Dead code:
  - `LOWER_IS_BETTER` in `src/trivium/generation/comparison.py:16`.
  - The `if json_mode ...: pass` block in `src/trivium/models/argo.py:143`.
  - The redundant branch in `src/trivium/evaluation/difficulty.py:30`.
  - `Feedback.overused_fact_ids` is populated but never read.
- `accuracy()` defaults to `max_words=12` and is never passed the config's `max_answer_words`.
- The `httpx` clients in `ArgoLLM` and `ArgoEmbedder` are never closed.
- An unknown Argo `env` raises a bare `KeyError` (`resolve_base`).
- `comparable_groups`, `DerivedGenerator.pairs()` and `two_hop_paths()` are recomputed every round (in both `available()` and `propose()`) against a static store; they should be cached.
- `perturb_number` can produce non-integer counts (3 becomes 4.5) or percentages over 100.
- The multi-hop leak check rejects short numeric answers that appear inside another number: "5" is found in "2015".
- The extraction prompt asks for a verbatim evidence quote, but nothing checks that the evidence appears in `chunk.text`. That would be a cheap deterministic guard against hallucinated quotes.
- `_quota_consistency` does not check that the quotas sum to `num_questions`; the "extreme" difficulty has no quota.
- `ModelSpec.merged` uses `model_copy(update=...)`, which skips validation and replaces `options` rather than merging it.
- `HashingEmbedder` silently ignores `embeddings.options`.
- Section ids `preamble` / `body` can collide with a heading that slugifies to the same string, giving duplicate chunk ids.
- Headings inside fenced code blocks (`# comment`) are treated as markdown headings by the parser.
- `Controller._maybe_multiple_choice` calls the private `StrategySelector._type_target`.
- By default the generator, quality evaluator, open-book model and large tier are all the same model, so there is self-evaluation bias. The README should recommend a different judge.

## 4. Tests

The three reproduced bugs need regression tests:

1. `match_answer("A", ["A", "..."], "choice")` is `True`, and the pipeline test should assert that some accepted multiple-choice item has "A" as its answer.
2. `_fmt(1000) == "1000"`, `_fmt(1200) == "1200"`, `_fmt(250000) == "250000"`.
3. `ask_llm` with a `fixed_answer` of `"12.5"` and an LLM answer of `"2.5"` yields `"12.5"`.

Also worth adding: grading negatives from section 2.1, and a test that a transient embedder failure does not abort the run.

## 5. Suggested order of work

1. Fix 1.1, 1.2, 1.3 with regression tests.
2. Tighten grading (2.1), since it feeds every accuracy-based metric.
3. Robustness: move the backoff sleep outside the semaphore, distinguish infrastructure errors from wrong answers, add embedder retries (2.2, 2.3).
4. CLI/config precedence and the username leak (2.5, 2.8).
5. Cleanup items in section 3.

## 6. Resolution (2026-09-19)

All items were addressed in the commit following this review; 78 tests pass (29 new in
`tests/test_review_regressions.py`), and a live Argo run confirmed an accepted multiple-choice
item with answer "A" and no per-user options in the output config.

| Item | Change |
|---|---|
| 1.1 | `normalize_answer` no longer strips stop-words from single-token answers; `choice` letters checked explicitly. |
| 1.2 | `_fmt` strips trailing zeros only when a decimal point is present. |
| 1.3 | `answers_match` is a contiguous-token-subsequence test; `ask_llm` always sets `answer = fixed_answer` and keeps the LLM phrasing only as an alias when equivalent. |
| 2.1 | Numeric path only for numeric-like references, units compared, hedges/negations (`or`, `not`, …) rejected, standalone-number tokenisation ("9" in "QM9X" is not a number), qualified-prefix rule for "DFT with the wB97X functional". |
| 2.2 | Backoff sleep moved outside the semaphore; `is_retryable()` stops retries on 4xx/auth/config errors. |
| 2.3 | Failed answering trials return `None` and are excluded (all-failed raises); `Evaluator.evaluate` converts `LLMError` into `evaluation_error` with `Evaluation.error` instead of aborting; OpenAI/Argo embedders retry via `with_retries`. |
| 2.4 | Controller's same-round re-check uses `Evaluator.is_duplicate` (judge + fact-overlap rule). |
| 2.5 | `--input/--topic` may come from the config file; precedence is CLI > env > YAML > defaults; `--provider` help lists `argo`. |
| 2.6 | `OpenAILLM` sends `max_completion_tokens` and no sampling parameters to `o*`/`gpt-5*`. |
| 2.7 | `ModelSpec.key()` includes `api_key_env`, `max_tokens`, `max_concurrency`, `options`; the response-cache key includes `max_tokens`. |
| 2.8 | `Config.public_dump()` redacts `options.user`/`api_key`/… in the embedded config. |
| 2.9 | `_split_long` returns offsets (no re-joining/searching); over-long sentences split at word boundaries. |
| 3 | Dead code removed (`LOWER_IS_BETTER`, Argo `pass` block, difficulty branch); `overused_fact_ids` now down-weights sampling; `accuracy()` gets `max_answer_words`; `aclose()` on LLMs/embedders/registry, called by `generate_dataset` and `extract`; unknown Argo env → `ArgoConfigError`; groups/pairs/paths cached per generator; perturbations keep integers integer and percentages ≤ 100; multi-hop leak check is token-bounded; extractor drops facts whose evidence is not a quote from the chunk; quota sum validated against `num_questions`; `ModelSpec.merged` validates and merges `options`; `HashingEmbedder` accepts `dim`; `preamble` id kept unique; fenced code blocks ignored by heading detection; `type_target` public; README recommends a different evaluator family. |

Deliberately not changed: no quota for `extreme` (it is a rejection category, not a target), and
the hashing embedder still recomputes locally on re-runs (deterministic and free).
