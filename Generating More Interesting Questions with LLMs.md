# Generating More Interesting Questions with LLMs

Strategies and resources for producing sharper, more realistic questions, for example for RAG evaluation sets and chatbot testing.

## Why AI-generated questions tend to be dull

- **Pull towards the median.** Models predict likely text, and preference tuning pushes them further towards safe, broadly acceptable output. Interesting questions sit in the less likely but still apt part of the distribution.
- **Answer leakage.** When a model writes questions from a document it can see, it tends to ask what the text plainly answers. Real users ask from what they *don't* know.
- **No stake or curiosity.** Good questions come from a goal, a confusion or a doubt. A model with no goal falls back on generic "what is X" questions.

## Strategies

### 1. Persona-driven generation

Describe a specific asker before generating:

- Role and expertise level (for example, a new user on their first allocation, or a seasoned HPC user moving from another facility)
- Goal (what they are trying to get done right now)
- Prior knowledge and misconceptions
- Frustrations or doubts ("my job sits in the queue forever and I don't know why")

Rotate through many personas to get more variety.

### 2. Hide the answer (the gap-driven approach)

Give the generator only a title, abstract or short summary, not the full text. Questions then come from real information gaps. Afterwards, check them against the full corpus and sort them into answerable, partly answerable and unanswerable. All three groups are useful for evaluation.

### 3. Multi-perspective interviews

Simulate a conversation: a persona asks, an "expert" grounded in the corpus answers, and the persona asks follow-ups. Follow-up questions are often the most interesting, because they build on what was just learned.

### 4. Question evolution

Start from simple seed questions and rewrite them step by step:

- Add constraints or conditions
- Ask for comparisons or trade-offs
- Turn them into multi-hop questions that need several sources
- Add counterfactuals ("what if the temperature were doubled?")
- Make them more abstract or more practical

### 5. Over-generate, then filter hard

Generate 10 to 50 times more questions than you need. Then filter using:

- A judge model that scores novelty, specificity, difficulty and realism
- Removal of near-duplicates with embeddings
- Diversity sampling (for example, clustering and picking from each cluster)
- Human review of a sample to calibrate the judge

### 6. Ground in real user behaviour

Seed generation with real questions from support logs, forum threads, mailing lists, GitHub issues or Q&A sites, and ask the model to write more questions *in that style*. Real phrasing, vagueness and mistakes make evaluation sets far more realistic.

### 7. Push the sampling

Use higher temperature for idea generation, explicitly ask for unusual or contrarian angles, and ban generic openers such as "What is...". You can also ask for questions an expert would find surprising or that would start a debate.

### 8. Use question taxonomies

Make sure the set covers several question types: factual, procedural ("how do I..."), diagnostic ("why does...fail"), comparative, hypothetical, evaluative ("is X trustworthy for Y?") and meta questions ("what's the limitation of..."). Bloom's taxonomy is a handy checklist for levels of thinking.

## Resources

These are listed from memory. Please check the current versions and links.

### Papers

- **STORM** (Shao et al., Stanford, 2024): *Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models*, arXiv 2402.14207. Perspective-guided question asking through simulated conversations with an expert.
- **Persona Hub** (Tencent AI Lab, 2024): *Scaling Synthetic Data Creation with 1,000,000,000 Personas*, arXiv 2406.20094. Shows that varying personas greatly increases the diversity of synthetic data.
- **WizardLM / Evol-Instruct** (2023): arXiv 2304.12244. Makes instructions progressively more complex through "evolution" steps, which can be adapted to questions.

### Libraries

- **Ragas**: its test-set generation for RAG supports personas and different query synthesisers (single-hop, multi-hop, specific and abstract).
- **DeepEval (Confident AI)**: its Synthesizer generates "goldens" from documents and applies evolutions to raise complexity.

### Related ideas worth exploring

- LLM-as-judge for rating question quality
- Embedding-based deduplication and diversity selection
- Adversarial or red-team question generation to stress-test RAG systems

## Suggested pipeline for Ask ALCF

The goal is a hard, stable benchmark that tracks Ask ALCF's progress from one version to the next.

1. **Define ALCF user personas.** Examples: a first-time user with a new allocation; a PI managing project hours and team members; an experienced user moving from another facility or between ALCF systems; an AI/ML user setting up conda, containers or distributed training; a data-heavy user worried about file systems and transfers; a user who has hit an error and is frustrated.
2. **Generate questions from gaps.** Show the generator only page titles or the user guide's table of contents, not full pages, and ask what each persona would actually need to know.
3. **Mine real questions.** Seed with anonymised support tickets, user Slack or mailing-list threads, and training-event Q&A, then generate more in the same style.
4. **Target the hard categories:**
   - Multi-hop questions spanning several pages (job script plus modules plus file system plus queue policy)
   - Troubleshooting from an error message or symptom
   - Version or system-specific questions where outdated pages could mislead
   - Ambiguous questions where the right move is to ask for clarification
   - Out-of-scope or unanswerable questions where the bot should say it doesn't know, or point to support
   - Questions phrased with user jargon, typos or the wrong terminology
5. **Evolve the seeds.** Add constraints ("on a debug queue", "with 64 nodes", "using MPI and GPUs"), comparisons, and "what if" variations.
6. **Over-generate and filter.** Remove duplicates, score with a judge for difficulty and realism, and sample across personas and categories.
7. **Label against the docs.** Mark each question as answerable, partly answerable or unanswerable, with the reference pages and a reference answer.
8. **Human review.** Have ALCF support staff check a sample. They know which questions users really struggle with.
9. **Keep a held-out set and refresh it.** Freeze one core set for comparisons over time, and add new questions as the docs and systems change.
