# Evaluation

The machinery lives in a separate package: **[eval-harness]
(https://github.com/mukundjaiswal/eval-harness)**, shared with my classification
and summarization projects. This document covers what is specific to this agent.

## The split

| Concern | Where it lives |
|---|---|
| Dataset format and loading | `eval-harness` |
| Metric registry, direction, aggregation | `eval-harness` |
| Seeded repetition, latency capture | `eval-harness` |
| Report format and the regression gate | `eval-harness` |
| **Which metrics this agent is judged on** | `src/agentic_rag/eval_metrics.py` |
| **How one question becomes an observation** | `cli.py`, the `task` closure |

The harness takes a callable, a dataset and metric names. It has no concept of
groundedness, retrieval or a repair budget, and it should not: those are
properties of this agent, and a classifier sharing the same harness has no use
for them.

Two things this project keeps that the shared package deliberately does not
impose:

- **The dataset keeps its `question` key.** `load_cases(path, input_key=
  "question")` reads the existing format rather than requiring every dataset in
  the portfolio to be rewritten around a generic `input` field.
- **Domain metrics are registered at import.** `register_agent_metrics()` is
  idempotent, so importing it from the CLI and from a notebook in the same
  process is safe.

## What is missing

| Gap | Why it matters |
|---|---|
| No committed dataset | The corpus is not mine to redistribute, so the gate cannot run in CI yet |
| Judges uncalibrated | Agreement with a human rater is unmeasured, so scores report the judge's consistency, not correctness |
| Retrieval not scored separately | A bad answer could be a retrieval miss or a generation miss; current metrics cannot tell you which |
| No slice-level breakdown | Aggregates hide which kind of question regressed. `EvalCase.slice()` reads the metadata for this; nothing groups by it yet |
| No token or cost accounting | Latency is measured; spend is not, and a repair loop is where spend hides |

## Why multiple seeds

A single run over a small dataset cannot distinguish a real improvement from
noise. The harness reports mean and standard deviation across seeds, and a
single-seed report says so in its own summary rather than presenting one number
as settled.

Reporting a mean without a spread is how a two-point difference gets presented
as progress.

## Why `mean_repairs` is worth watching

It is a leading indicator. When retrieval quality degrades — a re-index, a
chunking change, a model swap — the repair loop absorbs the damage first. Answer
quality holds while repairs per question climb, and latency and cost climb with
them. By the time `mean_groundedness` moves, the problem has been there a while.

## Calibrating the judges

The next real step, in order:

1. Sample 100 responses across the score range.
2. Label them by hand for groundedness.
3. Compute agreement (Cohen's kappa) between the judge and the labels.
4. If agreement is poor, the prompt is the problem, not the threshold — revise
   the rubric and re-measure.
5. Commit the labelled sample as a fixture so the judge itself is regression
   tested when the prompt or the model changes.

Until that exists, every number in a report is the judge's opinion of itself.
