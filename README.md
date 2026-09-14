[![CI](https://github.com/mukundjaiswal/agentic-rag-assistant-nutrition/actions/workflows/ci.yml/badge.svg)](https://github.com/mukundjaiswal/agentic-rag-assistant-nutrition/actions/workflows/ci.yml)

# Agentic RAG Assistant — Nutrition

A retrieval-augmented assistant built as an explicit **state machine** rather
than a single retrieve-then-generate call. The agent judges its own output,
decides whether it is good enough to return, and repairs it along one of two
different paths under a bounded budget.

The demo corpus is nutrition and dietary reference material, which is a domain
where a confidently wrong answer does real harm — a good reason to build the
quality gates first and the chat interface second. Nothing under `src/` knows
anything about nutrition: the domain lives entirely in the indexed corpus, so
pointing it at a different one is an `ingest` command, not a code change.

This is a personal project. It is not derived from any employer's systems, data
or code.

---

## The idea

Most RAG examples stop at *retrieve, generate, return*. The interesting
engineering starts after that, at the question **"should this answer ship?"**

```
        expand_query
             |
        retrieve_context  <--------------------------+
             |                                       |
        craft_response                               |
             |                                       |
      score_groundedness <-----------+          refine_query
        |            |               |               ^
      pass          fail        refine_response      |
        |            |               |               |
        v            +---------------+               |
      check_precision ------------------- fail ------+
             |
           pass -> return     (budget spent -> decline honestly)
```

**Two failure modes, two different repairs.** That distinction is the design:

| Symptom | Diagnosis | Repair |
|---|---|---|
| Low **groundedness** | The answer drifted from the retrieved context. Retrieval was fine; generation was not. | Rewrite the **response** against the same context. |
| Low **precision** | The answer is faithful to the wrong material. | Rewrite the **query** and retrieve again. |

Repairing the wrong one is worse than not repairing at all: rewriting a response
that was never going to answer the question just spends tokens producing a more
polished irrelevance.

**The budget is a feature.** A self-correcting loop without a cap is an
unbounded cost and latency risk. When the budget is spent, the agent says it
could not answer confidently. In a support setting that beats a fourth rewrite
of a wrong answer.

**An unreadable judge is not a pass.** If a judge returns prose instead of a
score, the run routes to repair. A quality gate that fails open is not a gate,
and this is the single most common way a harness quietly stops working.

---

## Quick start

```bash
git clone https://github.com/mukundjaiswal/agentic-rag-assistant-nutrition.git
cd agentic-rag-assistant-nutrition

python -m venv .venv && source .venv/bin/activate

# The shared evaluation harness is a sibling repo, not yet on PyPI.
pip install "eval-harness @ git+https://github.com/mukundjaiswal/eval-harness.git"
pip install -e ".[rag,ui,dev]"

cp .env.example .env          # fill in your keys
agentic-rag config            # verify what loaded; secrets are masked

agentic-rag ingest ./data/corpus
agentic-rag ask "Does calcium interfere with iron absorption?" --show-trace
streamlit run src/agentic_rag/ui/streamlit_app.py
```

Run the tests without any of that — they need no key and no network:

```bash
pytest
```

---

## Opening it in IntelliJ IDEA / PyCharm

1. **File -> Open** and select the project root (the folder holding
   `pyproject.toml`). Open the folder, not a single file.
2. **File -> Project Structure -> SDKs**: add a Python SDK. Either point it at
   an existing `.venv`, or create one — IntelliJ offers this when it sees
   `pyproject.toml`.
3. In the IDE terminal, install the project in editable mode:
   ```bash
   pip install -e ".[rag,ui,dev]"
   ```
   This is what makes `from agentic_rag...` resolve everywhere. With a `src/`
   layout, an editable install is the correct fix; marking `src` as a Sources
   Root by hand works too but only inside the IDE, so the test runner and the
   CLI would still disagree with it.
4. **Settings -> Tools -> Python Integrated Tools**: set the default test runner
   to **pytest**. Right-click `tests/` -> Run.
5. Optional: **Settings -> Tools -> Ruff** (plugin) for inline lint, and enable
   mypy via the terminal (`make typecheck`) or the Mypy plugin.

The `.idea/` directory is gitignored, so your IDE settings stay yours.

---

## Layout

```
src/agentic_rag/
  settings.py          typed, validated config; secrets held as SecretStr
  interfaces.py        Protocols for every external dependency
  exceptions.py        one base class, so callers never catch Exception
  logging_config.py    text or JSON logs, correlated by run_id

  agent/
    state.py           the state carried between nodes
    prompts.py         every template, in one place
    judges.py          LLM-as-judge scoring, with strict and lenient modes
    routing.py         pure decision functions  <- the heart of it
    nodes.py           one responsibility per node
    dependencies.py    the only place vendor clients get constructed
    graph.py           wiring and the runner

  ingestion/           parse -> chunk -> augment -> index
  retrieval/           Chroma adapter + an in-memory fake
  llm/                 provider adapter + a scripted fake
  eval_metrics.py      this agent's metrics, registered with eval-harness
  memory/              bounded session window
  safety/              input filter (disabled; see below)
  cli.py               ingest / ask / evaluate
  ui/                  Streamlit chat

tests/
  unit/                fast, offline, no keys
  integration/         the compiled graph, driven by fakes
```

### Why Protocols instead of base classes

`interfaces.py` defines `ChatModel`, `DocumentRetriever` and `Judge` as
`typing.Protocol`. Nothing inherits from them; a class satisfies one by having
the right shape. The agent depends only on those protocols and never imports a
vendor SDK, so:

- the entire graph runs in tests against `ScriptedChatModel` and
  `InMemoryRetriever`, with no network, no key and no token spend;
- swapping the model provider or the vector store is an edit to one adapter.

`dependencies.py` is the single place a real client is constructed.

### Why routing is a separate module of pure functions

Routing is the part of an agent that fails silently. A misordered condition
sends every run down the repair path and the only symptom is a larger invoice at
the end of the month. Pulling those decisions into pure functions with no I/O
makes them a unit test — see `tests/unit/test_routing.py`, including the case
where a passing score at the budget ceiling must still be honoured rather than
discarded.

---

## Evaluation

Scoring, aggregation across seeds, the report format and the regression gate all
come from **[eval-harness](https://github.com/mukundjaiswal/eval-harness)**, a
separate package shared with my classification and summarization projects.

That split is deliberate and it is not cosmetic. The harness knows nothing about
agents: it takes a callable, a dataset and a set of metrics. What belongs to
*this* project is one file, `src/agentic_rag/eval_metrics.py`, declaring the five
things this agent is judged on and which direction is an improvement for each.
A classifier has no use for `groundedness`; a harness that shipped it would be
pretending otherwise.

```bash
# establish a baseline
agentic-rag evaluate --dataset evaluation/datasets/v1.jsonl \
  --out evaluation/baselines/v1.json

# later: compare across seeds, exit non-zero on a regression
agentic-rag evaluate --dataset evaluation/datasets/v1.jsonl \
  --baseline evaluation/baselines/v1.json --tolerance 0.02 --seeds 0,1,2
```

Metrics this agent registers:

| Metric | Direction | What it tells you |
|---|---|---|
| `mean_groundedness` | higher | Answers supported by the retrieved context |
| `mean_precision` | higher | Answers that address the question asked |
| `mean_repairs` | lower | Repairs per question. **Leading indicator** — this rises before answer quality falls, because the loop absorbs the damage first |
| `exhaustion_rate` | lower | Share of questions the agent declined |
| `unparsed_judge_rate` | lower | Harness health. If non-zero, every quality number above it is measured on a smaller sample than the case count |
| `p50` / `p95` latency | lower | Supplied by the harness. The tail is what users feel; the mean hides the runs that repaired three times |

Direction is declared at registration and travels on the report, so a baseline
still compares correctly after the code that produced it has changed.

The dataset keeps its documented `question` key — `load_cases(..., input_key=
"question")` — rather than being rewritten to suit the shared package.

See `docs/evaluation.md`.

## What this does not do

Written down because the gaps are the interesting part, and because a portfolio
that implies more rigor than it has gets found out in the first technical
conversation.

- **The judges are not calibrated.** Agreement with a human rater is unmeasured,
  so a score currently reports the judge's own consistency, not correctness.
  Calibrating against a labelled sample is the first thing worth doing.
- **The safety filter is off.** `safety/guardrails.py` holds the interface
  behind a flag, disabled. The hosted moderation model the original
  implementation called was retired by the provider, and a filter that returns
  "allowed" because the request failed is worse than no filter: it manufactures
  confidence. Enabling it without a backend raises rather than failing open.
- **No regression gate in CI yet.** `RegressionCheck` and the `--baseline` flag
  exist and are tested; the workflow step that calls them is commented out until
  a dataset is committed.
- **Retrieval is not evaluated separately from answers.** A bad answer could be
  a retrieval failure or a generation failure, and the current metrics do not
  separate them.
- **No cost accounting.** Latency is measured; tokens are not.

---

## Development

```bash
make install-dev    # editable install + pre-commit hooks
make check          # ruff, mypy --strict, pytest
make cov            # coverage report
```

CI runs lint, format check, `mypy --strict` and the full suite on 3.11 and 3.12.

## License

MIT. See [LICENSE](LICENSE).
