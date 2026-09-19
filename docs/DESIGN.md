# Adaptive Trivia and RAG Question Generation System

## 1. Purpose

The goal of this project is to automatically generate high-quality, objectively gradable questions from either:

1. a user-specified **topic**, or
2. one or more **documents/directories** provided by the user.

The generated questions should be useful for two related applications:

* human trivia or quiz activities;
* evaluation of retrieval-augmented generation (RAG) systems.

Questions should have concise, deterministic answers whenever possible:

* short answer: one word or short phrase;
* numerical answer;
* True/False;
* multiple choice.

The distinguishing feature of the system is that question generation is **adaptive**. Candidate questions are evaluated on multiple dimensions, and those evaluation results are used to guide subsequent generation toward:

* higher-quality questions;
* less redundant questions;
* better topic/document coverage;
* questions that genuinely require the source material;
* questions of requested or empirically calibrated difficulty.

The system should produce a JSON or JSONL dataset containing each accepted question, reference answer, source evidence, metadata, and evaluation results.

---

# 2. Design Principles

The system should follow several principles.

### 2.1 Generate more candidates than are needed

If the user requests 100 questions, the system should not simply ask an LLM to generate 100 questions.

Instead:

```text
generate candidates
      ↓
evaluate
      ↓
reject weak questions
      ↓
analyze remaining gaps
      ↓
generate targeted candidates
      ↓
repeat until quota is reached
```

The final 100 questions may come from 200–500 generated candidates.

---

### 2.2 Facts are the canonical knowledge representation

Documents should first be converted into structured facts with provenance.

A knowledge graph can then be constructed from these facts.

```text
documents
   ↓
chunks
   ↓
facts
   ↓
knowledge graph
   ↓
question candidates
```

The fact representation, rather than the knowledge graph itself, should be the primary source of truth.

---

### 2.3 Every question must be grounded

Every accepted question should have one or more supporting facts and evidence spans.

For example:

```json
{
  "question": "Which exchange-correlation functional was used?",
  "answer": "PBE",
  "evidence": [
    {
      "document": "paper.pdf",
      "chunk_id": "chunk_42",
      "text": "Calculations were performed using the PBE functional."
    }
  ]
}
```

This enables:

* automatic verification;
* retrieval evaluation;
* debugging;
* identification of hallucinated questions.

---

### 2.4 Difficulty should be measured rather than guessed

The generator may attempt to create easy or hard questions, but the final difficulty label should primarily be based on empirical evaluation.

For example:

```text
small model + evidence     correct
medium model + evidence    correct
large model + evidence     correct

→ easy
```

versus

```text
small model + evidence     wrong
medium model + evidence    wrong
large model + evidence     correct

→ hard
```

---

### 2.5 Document dependence should be measured

A strong RAG evaluation question should ideally be difficult to answer without the provided resources but easy to answer when the correct evidence is available.

Define:

$$
D = P(\mathrm{correct}\mid\mathrm{evidence})
    - P(\mathrm{correct}\mid\mathrm{closed\ book})
$$

A question with:

```text
Closed book: 0/5
Open book:   5/5
```

has:

```text
document_dependence = 1.0
```

This is highly desirable for a RAG benchmark.

---

# 3. User Interface

A command-line interface can initially be sufficient.

Example:

```bash
qgen generate \
    --input ./papers/ \
    --num-questions 100 \
    --output questions.json
```

For a topic:

```bash
qgen generate \
    --topic "history of artificial intelligence" \
    --num-questions 50 \
    --output ai_history.json
```

Optional controls:

```bash
--easy 30
--medium 40
--hard 30

--question-types short_answer,true_false,multiple_choice

--max-answer-words 8

--require-document-dependence

--multi-hop-ratio 0.25

--cross-document-ratio 0.10

--seed 1234
```

If `--num-questions` is omitted, the system determines a reasonable target automatically.

---

# 4. Automatic Determination of Question Count

The system should estimate how many high-quality questions the material can support.

Factors include:

* total document length;
* number of extracted facts;
* number of unique entities;
* number of relationships;
* number of document sections;
* number of potentially useful graph paths;
* amount of redundancy;
* percentage of candidate facts that pass verification.

A simple initial heuristic could be:

```python
target = min(
    max_questions,
    int(
        0.4 * atomic_facts
        + 0.2 * relational_facts
        + 0.1 * useful_graph_paths
    )
)
```

The exact heuristic should eventually be learned from observed generation statistics.

The system should also stop early if additional generation repeatedly fails to produce sufficiently novel questions.

---

# 5. High-Level Architecture

```text
                   ┌─────────────────────┐
                   │ User Input          │
                   │ topic / documents   │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Document Processor  │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Fact Extractor      │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Fact Validator      │
                   └──────────┬──────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
       ┌──────────────────┐      ┌──────────────────┐
       │ Fact Store       │      │ Knowledge Graph  │
       └─────────┬────────┘      └─────────┬────────┘
                 └────────────┬────────────┘
                              ▼
                   ┌─────────────────────┐
                   │ Candidate Generator │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Question Evaluator  │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Difficulty Tester   │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Coverage Analyzer   │
                   └──────────┬──────────┘
                              │
                   feedback ──┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Final Question Bank │
                   └─────────────────────┘
```

---

# 6. Major Components

## 6.1 Input Manager

Supported inputs should eventually include:

```text
.txt
.md
.html
.pdf
.docx
directory trees
URLs
```

The initial MVP could support:

```text
.txt
.md
.pdf
```

For topic-based generation, a separate resource acquisition step is required.

Possible future implementations:

```text
topic
 ↓
web search / Wikipedia / supplied search backend
 ↓
source corpus
```

Topic-derived datasets should retain the URLs and timestamps of all sources.

---

# 6.2 Document Processor

Responsibilities:

* parse documents;
* remove headers/footer noise;
* identify headings and sections;
* create chunks;
* preserve document hierarchy;
* assign stable IDs.

Example:

```json
{
  "document_id": "doc_003",
  "section_id": "methods",
  "chunk_id": "doc_003_methods_017",
  "text": "...",
  "start_char": 12331,
  "end_char": 14109
}
```

Chunk size should be configurable.

Semantic or structure-aware chunking should be preferred over arbitrary fixed token boundaries.

---

# 7. Fact Extraction

Each chunk is passed to an LLM-based structured extractor.

Example internal representation:

```json
{
  "id": "fact_1287",
  "subject": "MACE",
  "predicate": "mean_absolute_error",
  "object": {
    "value": 0.12,
    "unit": "eV"
  },
  "qualifiers": {
    "dataset": "test set"
  },
  "fact_type": "measurement",
  "explicit": true,
  "document_id": "doc_003",
  "chunk_id": "doc_003_results_017",
  "evidence": "MACE obtained an MAE of 0.12 eV.",
  "confidence": 0.96
}
```

Recommended fact types:

```text
attribute
definition
quantity
event
relationship
comparison
classification
temporal
location
causal
procedure
result
```

---

# 8. Fact Validation

Facts should be validated before question generation.

Validation should test:

### Evidence entailment

Does the evidence actually support the extracted fact?

### Atomicity

Does the fact represent one claim rather than several unrelated claims?

### Explicitness

Is the fact directly stated or inferred?

### Ambiguity

Are subject and object sufficiently clear?

### Referential resolution

Pronouns and references should be resolved.

For example:

```text
"It achieved 93% accuracy."
```

should not become a fact unless `it` can be confidently mapped to an entity.

Each fact receives:

```json
{
  "fact_quality": {
    "entailment": 0.99,
    "atomicity": 0.94,
    "ambiguity": 0.02,
    "overall": 0.96
  }
}
```

Low-quality facts are discarded.

---

# 9. Knowledge Graph

Validated facts can be converted into a graph.

Nodes may represent:

```text
entities
methods
people
locations
experiments
datasets
software
quantities
events
concepts
```

Edges represent relationships.

Each graph edge must retain its provenance:

```python
graph.add_edge(
    "MACE",
    "0.12 eV",
    predicate="mean_absolute_error",
    fact_id="fact_1287"
)
```

Initial implementation:

```text
NetworkX
```

Future implementations may support:

```text
Neo4j
Ragas KnowledgeGraph
LlamaIndex PropertyGraphIndex
```

The core API should remain backend-independent.

---

# 10. Question Generation Strategies

The system should contain multiple specialized generators rather than one generic prompt.

## 10.1 Atomic Fact Questions

Generated directly from a single fact.

Example:

```text
Fact:
The simulation temperature was 450 K.

Question:
What temperature was used?

Answer:
450 K
```

---

## 10.2 Relationship Questions

Combine an entity with one of its relations.

```text
Fact:
Experiment 4 → catalyst → Ru

Question:
Which catalyst was used in Experiment 4?

Answer:
Ru
```

---

## 10.3 Comparison Questions

Generated from related facts.

```text
MACE MAE = 0.12 eV
xTB MAE  = 0.21 eV
```

Question:

```text
Which method had the lower MAE?

Answer:
MACE
```

---

# 10.4 Derived Questions

Require deterministic operations.

Examples:

```text
difference
ratio
maximum
minimum
ordering
counting
```

Example:

```text
Initial error = 0.31 eV
Final error   = 0.18 eV
```

Question:

```text
By how much did the error decrease?

Answer:
0.13 eV
```

Derived facts should explicitly record their dependencies.

---

# 10.5 Multi-Hop Questions

Generated from graph paths.

Example:

```text
Model A
 → trained_on → Dataset X
 → created_by → Institution Y
```

Question:

```text
Which institution created the dataset used to train Model A?

Answer:
Institution Y
```

The graph path should be stored with the question.

---

# 10.6 Cross-Document Questions

Require facts from more than one source document.

These are especially valuable for testing RAG systems over collections.

---

# 10.7 True/False Questions

Prefer modifying validated facts rather than allowing unconstrained fabrication.

True question:

```text
MACE achieved an MAE of 0.12 eV.
```

False question:

```text
MACE achieved an MAE of 0.21 eV.
```

False statements must be checked to ensure that the modified version is genuinely contradicted rather than merely absent.

---

# 10.8 Multiple-Choice Questions

Generate and validate the question/answer pair first.

Then separately generate distractors.

```text
Question
   ↓
Correct answer
   ↓
Validate
   ↓
Generate distractors
   ↓
Validate each distractor
```

Distractors should:

* have the same semantic type as the answer;
* be plausible;
* not also be correct;
* preferably come from nearby entities or related facts.

---

# 11. Candidate Question Representation

```json
{
  "id": "q_00421",
  "question": "Which method had the lowest mean absolute error?",
  "answer": "MACE",
  "answer_type": "short_answer",
  "accepted_answers": [
    "MACE"
  ],
  "source_facts": [
    "fact_1287",
    "fact_1288"
  ],
  "source_chunks": [
    "doc3_results_17"
  ],
  "generation_strategy": "comparison",
  "generation_round": 4
}
```

---

# 12. Question Quality Evaluation

This is a central component of the system.

The evaluator should not simply answer:

```text
Is this a good question?
```

Instead it should produce multiple independent metrics.

---

## 12.1 Answerability

Can the question be answered from the provided evidence?

```text
answerability ∈ [0,1]
```

---

## 12.2 Evidence Entailment

Does the provided evidence actually imply the reference answer?

```text
evidence_entailment ∈ [0,1]
```

This should be one of the strongest acceptance requirements.

---

## 12.3 Ambiguity

Could multiple answers reasonably be considered correct?

```text
ambiguity ∈ [0,1]
```

Lower is better.

---

## 12.4 Answer Conciseness

Is the answer compatible with deterministic grading?

```text
answer_conciseness ∈ [0,1]
```

---

## 12.5 Question Clarity

Does the question contain enough context to identify what is being asked without introducing unnecessary complexity?

```text
clarity ∈ [0,1]
```

---

## 12.6 Independence from Question Wording

Does the question accidentally reveal the answer?

Example of poor question:

```text
Which PBE functional was used in the PBE calculations?
```

Metric:

```text
answer_leakage ∈ [0,1]
```

Lower is better.

---

## 12.7 Interestingness

For trivia-oriented use:

```text
interestingness ∈ [0,1]
```

This is necessarily more subjective and should have less weight for RAG evaluation.

---

## 12.8 Specificity

Avoid questions that are too broad.

Bad:

```text
What did the authors find?
```

Good:

```text
Which model achieved the lowest MAE?
```

---

# 13. Closed-Book Evaluation

A strong model is asked the question without access to the source.

Run multiple trials if practical.

Example:

```text
5 trials
1 correct
```

Then:

```text
closed_book_accuracy = 0.2
```

A lower value is generally preferable for RAG evaluation.

This should not automatically reject the question unless the user requests strict RAG mode.

---

# 14. Open-Book Evaluation

The evaluator receives only the evidence associated with the question.

Example:

```text
large model + gold evidence
```

Run one or more trials.

```text
open_book_accuracy = 1.0
```

Questions for which a strong model cannot answer reliably given the correct evidence should normally be rejected.

---

# 15. Document Dependence

Compute:

$$
\mathrm{document\_dependence}
=
\mathrm{open\_book\_accuracy}
-
\mathrm{closed\_book\_accuracy}
$$

Example:

```text
open_book_accuracy   = 1.0
closed_book_accuracy = 0.2

document_dependence = 0.8
```

This should be a key RAG-benchmark metric.

---

# 16. Difficulty Calibration

Difficulty should be empirical.

The system can define several evaluator models:

```text
small
medium
large
```

Each receives the gold evidence.

Example:

```json
{
  "small_model_accuracy": 0.2,
  "medium_model_accuracy": 0.6,
  "large_model_accuracy": 1.0
}
```

Possible initial thresholds:

```text
Easy:
small ≥ 0.8

Medium:
small < 0.8
medium ≥ 0.8

Hard:
medium < 0.8
large ≥ 0.8

Invalid/Extreme:
large < 0.8
```

These thresholds should be configurable.

---

# 17. Separate Retrieval and Reasoning Difficulty

Difficulty should eventually be represented along multiple axes.

```text
answering_difficulty
retrieval_difficulty
reasoning_depth
```

For example:

```json
{
  "answering_difficulty": "easy",
  "retrieval_difficulty": "hard",
  "reasoning_depth": 1
}
```

Retrieval difficulty can be estimated using actual retrieval experiments.

For example:

```text
retrieve top-1
retrieve top-3
retrieve top-5
retrieve top-10
```

If the gold evidence only appears at rank 9:

```text
retrieval_difficulty = hard
```

---

# 18. Redundancy Evaluation

Questions should not simply paraphrase existing accepted questions.

Compute similarity using:

* embedding similarity;
* overlap in supporting facts;
* LLM semantic duplicate judgment.

Example:

```text
semantic_similarity_to_existing = 0.93
```

Such a question should normally be rejected.

Fact overlap can also reveal redundant questions:

```text
Q1 → fact_123
Q2 → fact_123
Q3 → fact_123
```

Even if the wording differs, coverage is poor.

---

# 19. Coverage Metrics

The generator should track:

```text
document coverage
section coverage
fact coverage
entity coverage
relation coverage
question-type distribution
difficulty distribution
generation-strategy distribution
```

Example:

```json
{
  "coverage": {
    "facts_used": 83,
    "total_valid_facts": 342,
    "fact_coverage": 0.243,
    "sections_used": 12,
    "total_sections": 14
  }
}
```

Coverage statistics guide subsequent generation rounds.

---

# 20. Adaptive Generation Loop

This is the core algorithm.

```python
accepted = []

while not target_reached():

    status = analyze_question_bank(
        accepted_questions=accepted,
        facts=facts,
        graph=graph,
        requested_distribution=config
    )

    strategy = choose_generation_strategy(status)

    candidates = generate_candidates(
        strategy=strategy,
        facts=facts,
        graph=graph,
        accepted_questions=accepted,
        feedback=status,
        batch_size=config.batch_size
    )

    for candidate in candidates:

        scores = evaluate(candidate)

        if passes_thresholds(scores):
            accepted.append(candidate)

    if convergence_detected():
        break
```

The next generation round should be explicitly informed by previous failures.

---

# 21. Feedback to the Generator

Suppose the system currently contains:

```text
Easy: 38 / 40
Hard: 11 / 30

Atomic:        overrepresented
Multi-hop:     underrepresented
Section 4:     underrepresented
Section 7:     no questions
```

The next generator request should look conceptually like:

```text
Generate hard candidate questions.

Current deficiencies:
- prioritize Sections 4 and 7;
- avoid atomic fact questions;
- prioritize two-hop and comparison questions;
- avoid facts already used by accepted questions;
- previous rejected candidates were too easy in closed-book testing.

Generate 8 candidates.
```

This is much more useful than repeating the original generic prompt.

---

# 22. Learning From Rejected Questions

Rejected questions should not simply disappear.

Record:

```text
reason_for_rejection
```

Examples:

```text
closed_book_too_easy
ambiguous
unsupported_answer
duplicate
answer_too_long
large_model_failed_open_book
bad_distractors
```

Aggregate statistics can then influence future generation.

Example:

```text
47% of "definition" questions fail closed-book novelty.

→ temporarily reduce definition-question generation.
```

Or:

```text
comparison questions have a 72% acceptance rate
and produce substantially more hard questions.

→ increase comparison-question sampling.
```

Eventually the system could learn generation-strategy weights automatically.

---

# 23. Strategy Selection

Initial implementation can use heuristics.

For example:

```python
strategy_weights = {
    "atomic": 0.25,
    "relationship": 0.25,
    "comparison": 0.20,
    "derived": 0.10,
    "multi_hop": 0.15,
    "cross_document": 0.05
}
```

The weights can adapt based on:

```text
acceptance rate
difficulty yield
coverage contribution
duplicate rate
closed-book failure rate
```

Conceptually:

$$
W_s \propto
\frac{
\text{useful accepted questions from strategy }s
}{
\text{candidate questions generated by strategy }s
}
$$

More sophisticated approaches could later use:

* multi-armed bandits;
* Bayesian optimization;
* reinforcement learning.

The first version should remain heuristic and interpretable.

---

# 24. Composite Quality Score

Individual metrics should always be preserved.

A composite score can additionally simplify ranking.

Example:

$$
Q =
0.25E +
0.20A +
0.15C +
0.15D +
0.10N +
0.10S +
0.05I
$$

where:

```text
E = evidence entailment
A = answerability
C = clarity
D = document dependence
N = novelty/non-redundancy
S = specificity
I = interestingness
```

The exact weights depend on mode.

### RAG mode

Increase:

```text
document dependence
evidence entailment
retrieval diversity
```

### Trivia mode

Increase:

```text
interestingness
clarity
difficulty balance
```

---

# 25. Question Acceptance

Acceptance should combine hard constraints and soft scoring.

Example:

```python
if evidence_entailment < 0.9:
    reject()

if answerability < 0.9:
    reject()

if ambiguity > 0.2:
    reject()

if open_book_accuracy < 0.8:
    reject()

if duplicate_similarity > 0.90:
    reject()

if total_quality_score < 0.75:
    reject()
```

Some metrics, such as interestingness, should affect ranking but not necessarily rejection.

---

# 26. Output JSON Schema

Recommended output structure:

```json
{
  "dataset": {
    "created_at": "2026-09-17T23:00:00",
    "source_type": "documents",
    "requested_questions": 100,
    "generated_candidates": 287,
    "accepted_questions": 100
  },

  "questions": [
    {
      "id": "q_0001",

      "question": "Which model achieved the lowest MAE?",

      "type": "short_answer",

      "answer": "MACE",

      "accepted_answers": [
        "MACE"
      ],

      "difficulty": "medium",

      "generation": {
        "strategy": "comparison",
        "round": 3,
        "source_fact_ids": [
          "fact_1287",
          "fact_1288"
        ]
      },

      "evidence": [
        {
          "document_id": "paper1",
          "section": "Results",
          "chunk_id": "chunk_42",
          "text": "MACE obtained an MAE of 0.12 eV while xTB obtained 0.21 eV."
        }
      ],

      "evaluation": {
        "answerability": 0.99,
        "evidence_entailment": 0.99,
        "clarity": 0.95,
        "specificity": 0.97,
        "ambiguity": 0.02,
        "answer_conciseness": 1.0,
        "answer_leakage": 0.01,
        "interestingness": 0.73,

        "closed_book_accuracy": 0.20,
        "open_book_accuracy": 1.00,
        "document_dependence": 0.80,

        "small_model_accuracy": 0.40,
        "medium_model_accuracy": 1.00,
        "large_model_accuracy": 1.00,

        "retrieval_difficulty": "medium",
        "reasoning_depth": 2,

        "max_similarity_to_existing_question": 0.31,

        "overall_quality": 0.94
      }
    }
  ]
}
```

---

# 27. JSONL Alternative

For large datasets, JSONL may be more convenient.

```text
metadata.json
questions.jsonl
facts.jsonl
documents.jsonl
```

This is probably preferable internally.

The user-facing command can still generate a single JSON file if requested.

---

# 28. Suggested Code Structure

```text
qgen/
│
├── cli.py
├── config.py
│
├── ingestion/
│   ├── loader.py
│   ├── parser.py
│   └── chunker.py
│
├── facts/
│   ├── models.py
│   ├── extractor.py
│   ├── validator.py
│   └── store.py
│
├── graph/
│   ├── builder.py
│   ├── backend.py
│   └── networkx_backend.py
│
├── generation/
│   ├── base.py
│   ├── atomic.py
│   ├── relationship.py
│   ├── comparison.py
│   ├── derived.py
│   ├── multihop.py
│   ├── true_false.py
│   └── multiple_choice.py
│
├── evaluation/
│   ├── quality.py
│   ├── entailment.py
│   ├── closed_book.py
│   ├── open_book.py
│   ├── difficulty.py
│   ├── redundancy.py
│   └── retrieval.py
│
├── orchestration/
│   ├── controller.py
│   ├── coverage.py
│   ├── strategy.py
│   └── stopping.py
│
├── models/
│   ├── llm.py
│   └── embeddings.py
│
└── output/
    ├── schema.py
    └── writer.py
```

---

# 29. Model Abstraction

Avoid tying the project to a single model provider.

Define something like:

```python
class LLM:
    def generate(
        self,
        messages,
        response_model=None,
        temperature=0
    ):
        ...
```

Model roles should be independently configurable:

```yaml
models:

  fact_extractor:
    model: strong-model

  question_generator:
    model: strong-model

  quality_evaluator:
    model: strong-model

  closed_book:
    model: strong-model

  difficulty_small:
    model: small-model

  difficulty_medium:
    model: medium-model

  difficulty_large:
    model: strong-model
```

This also makes experimental comparisons easy.

---

# 30. Reproducibility

Every question should preserve:

```text
generation model
model version
generation prompt version
temperature
seed if available
evaluation models
evaluator prompt version
timestamp
```

This is particularly important if the dataset is used scientifically.

---

# 31. Caching

Evaluation can become expensive.

Cache results using keys based on:

```text
model
prompt hash
question
context
settings
```

Fact extraction and closed-book testing should never need to be repeated unless their configuration changes.

---

# 32. Parallelism

Many operations are embarrassingly parallel:

```text
fact extraction across chunks
fact validation
candidate evaluation
closed-book trials
open-book trials
difficulty calibration
```

The orchestration layer should therefore support asynchronous execution.

---

# 33. Stopping Criteria

The generation loop should terminate when any of these occur:

### Requested quota achieved

```text
accepted >= N
```

### Difficulty quotas achieved

```text
easy >= requested_easy
medium >= requested_medium
hard >= requested_hard
```

### Exhaustion

No new acceptable questions have been generated for several rounds.

### Excessive redundancy

Candidate novelty falls below a threshold.

### Cost limit

Optional user-defined budget reached.

---

# 34. Dataset-Level Evaluation

The final dataset should have its own summary metrics.

Example:

```json
{
  "dataset_metrics": {

    "num_questions": 100,

    "difficulty_distribution": {
      "easy": 31,
      "medium": 39,
      "hard": 30
    },

    "type_distribution": {
      "short_answer": 52,
      "true_false": 20,
      "multiple_choice": 28
    },

    "average_quality": 0.91,

    "average_document_dependence": 0.78,

    "fact_coverage": 0.42,

    "section_coverage": 0.93,

    "duplicate_rate": 0.01,

    "candidate_acceptance_rate": 0.35
  }
}
```

These statistics provide a useful summary of the generated benchmark.

---

# 35. MVP

The first implementation should deliberately remain simple.

### Phase 1

Implement:

```text
document loading
chunking
fact extraction
fact validation

atomic questions
relationship questions
comparison questions

quality evaluation
closed-book evaluation
open-book evaluation

easy/hard calibration
duplicate detection

adaptive iterative generation

JSON output
```

Use:

```text
Pydantic
NetworkX
an embedding model
one or more configurable LLM APIs
```

Do not initially require:

```text
Neo4j
GraphRAG
Ragas
LlamaIndex
```

Adapters can be added later.

---

# 36. Phase 2

Add:

```text
multi-hop questions
cross-document questions
retrieval difficulty
multiple-choice distractor validation
derived numerical questions
topic-based source collection
visual dashboard
human evaluation
```

---

# 37. Phase 3

Use accumulated generation results to improve
