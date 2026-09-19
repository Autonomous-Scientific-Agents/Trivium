# Architecture: design sections → modules

| Design section | Module(s) |
|---|---|
| §3 CLI | `trivium/cli.py` |
| §4 automatic question count | `orchestration/stopping.py::estimate_target`, `Controller._target` |
| §6 input manager / topic acquisition | `ingestion/loader.py`, `topic/acquire.py` (Wikipedia MVP) |
| §6.2 document processor | `ingestion/parser.py` (sections, header/footer noise), `ingestion/chunker.py` |
| §7 fact extraction | `facts/extractor.py`, `facts/models.py` |
| §8 fact validation | `facts/validator.py` |
| §9 knowledge graph | `graph/backend.py` (protocol), `graph/networkx_backend.py`, `graph/builder.py` |
| §10.1–10.6 generators | `generation/{atomic,relationship,comparison,derived,multihop,cross_document}.py` |
| §10.7 true/false | `generation/true_false.py` (+ `evaluation/entailment.py::check_statement`) |
| §10.8 multiple choice | `generation/multiple_choice.py` (distractor generation + per-distractor validation) |
| §11 candidate representation | `generation/base.py::Candidate` |
| §12 quality metrics | `evaluation/quality.py` |
| §13–15 closed/open book, document dependence | `evaluation/answering.py`, `evaluation/evaluator.py` |
| §16 difficulty calibration | `evaluation/difficulty.py` |
| §17 retrieval difficulty | `evaluation/retrieval.py` |
| §18 redundancy | `evaluation/redundancy.py` |
| §19 coverage | `orchestration/coverage.py::analyze` |
| §20 adaptive loop | `orchestration/controller.py::Controller.loop` |
| §21–22 feedback / learning from rejections | `orchestration/coverage.py::build_feedback`, `generation/base.py::Feedback` |
| §23 strategy selection | `orchestration/strategy.py` |
| §24–25 composite score / acceptance | `evaluation/evaluator.py::Evaluator.composite`, thresholds in `config.py` |
| §26–27 output | `output/schema.py`, `output/writer.py` |
| §29 model abstraction | `models/llm.py`, `models/registry.py`, `models/embeddings.py` |
| §30 reproducibility | `EvaluatedCandidate.provenance`, `prompts.py::PROMPT_VERSION`, seeds in `GenerationContext.rng` |
| §31 caching | `models/cache.py` (SQLite, content-addressed, per trial) |
| §32 parallelism | `asyncio.gather` throughout; per-model `Semaphore(max_concurrency)` |
| §33 stopping criteria | `orchestration/stopping.py::should_stop` |
| §34 dataset-level metrics | `output/schema.py::dataset_metrics` |

## Evaluation order (cost-aware early exit)

1. free structural checks: answer length, answer-in-question, embedding + fact-overlap redundancy
2. quality rubric (1 LLM call)
3. open-book and closed-book trials (concurrent)
4. difficulty tiers (only for configured `difficulty_small` / `difficulty_medium`)
5. retrieval difficulty (embeddings only)
6. composite score

## Refinements relative to the design document

* Closed-book accuracy is chance-corrected for true/false (0.5) and multiple-choice (1/n)
  items before computing document dependence, so guessing does not count as prior knowledge.
* Multiple-choice items are produced by converting an *accepted* short-answer question and
  re-running the full evaluation on the converted item; if it fails, the short-answer form is kept.
* Without small/medium tiers, difficulty is a documented proxy (`difficulty_method: "proxy"`).
* A deterministic `mock` provider re-implements every task with regular expressions, so the whole
  loop is testable offline (`tests/`).

## Refinements from the first live runs (Argo, gpt41 family, 2026-09-19)

* Lower-is-better rubric fields (`ambiguity`, `answer_leakage`) carry their polarity in the JSON
  schema `description`; without it the validator scored every clear fact `ambiguity = 1.0`.
* Heading-only paragraphs are dropped before chunking (a "# Title" chunk produced a `section_title` fact).
* Pseudo-units (`year`, `count`, …) are kept on the fact but hidden by `FactObject.display()`
  ("2021 year" leaked into answers, statements and distractors).
* The structural answer-in-question check is skipped for `comparison` items, which necessarily name
  the compared entities; the LLM rubric still scores leakage for them.
* Integer references (years, counts) require exact numeric matches; the 1 % relative tolerance applies
  only to non-integer measured quantities (2023 was being accepted for 2021).
* True/false numeric perturbations shift years by a few years and scale other quantities by 0.6–1.6 at
  the original precision ("founded in 3886", "0.4375 eV" were the previous behaviour).
* Derived ratios are disabled for year-like operands (differences remain).
