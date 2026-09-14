# Architecture notes

Decisions worth the words, and what each one costs.

## The agent is a state machine, not a chain

A chain describes a fixed sequence. This agent needs to look at its own output
and choose what to do next, which is a graph with conditional edges. Expressing
that explicitly means the control flow is inspectable: `state["history"]` is the
node path an answer actually took, which is the first thing you want when an
answer is wrong.

**Cost:** more moving parts than a chain, and a dependency on a graph runtime.
Worth it only because the conditional behaviour is the point.

## Dependencies are injected, vendors are adapters

`agent/` imports `interfaces.py` and nothing from a vendor SDK. Real clients are
constructed in exactly one place, `agent/dependencies.py`.

This is what makes `tests/integration/` possible: the full compiled graph runs
against `ScriptedChatModel` and `InMemoryRetriever`, so the repair paths, the
budget ceiling and the unreadable-judge case are all exercised in CI with no key
and no token spend. A test that needs a paid API is a test that gets skipped and
then deleted.

**Cost:** one extra indirection between the agent and the model.

## Routing is pure

`agent/routing.py` has no I/O and no mutation. Everything it needs arrives as
arguments. This is where the non-obvious rules live:

- A score exactly at the threshold passes. Thresholds are inclusive.
- A **passing** score at the budget ceiling is still honoured. The budget stops
  further repair; it does not discard an answer that already cleared the gate.
  Getting this backwards throws away good answers on the last iteration.
- `None` never passes.

## An unreadable judge routes to repair

`judges.TemplateJudge.evaluate` returns `JudgeResult(score=None)` when the reply
is not a number in `[0, 1]`, including out-of-range values — a judge returning
`1.4` is malfunctioning, and clamping it to `1.0` would hide that.

There is a second mode, `evaluate_strict`, which raises. Offline evaluation uses
it, because there an unreadable judge is a bug to fix rather than a condition to
route around. Same judge, two call sites, different contract.

## Configuration is validated once, at load

`settings.py` uses `pydantic-settings` with bounded fields. A threshold outside
`[0, 1]` or a zero iteration cap fails immediately with a readable message,
rather than producing a graph that never terminates or one that terminates
before it can repair.

API keys are `SecretStr`. A settings object ends up in logs and tracebacks; keys
must not. Key presence is checked at call time, not load time, so `--help` and
the unit suite run without credentials.

## The safety filter is off, deliberately

The original implementation called a hosted moderation model that the provider
retired. The choices were: leave a call that now fails, delete the concept, or
keep the interface and flag it off.

Flagged off, with two properties that matter:

1. A disabled filter's verdict says `reason="filter_disabled"` and
   `was_screened is False`. A caller can distinguish "screened and allowed" from
   "not screened", which a bare `True` cannot express.
2. Enabling the flag without configuring a backend **raises**. The one thing
   that must never happen is silently failing open.

## Chroma, and when it stops being right

Chroma runs from a local directory with no service to stand up, which is the
right trade for a single-node assistant. It is the wrong trade as soon as the
index must be shared across processes or machines. The `DocumentRetriever`
protocol is what keeps that migration scoped to one module.

## Bounded session memory

An unbounded transcript grows the prompt every turn until it crowds out the
retrieved context. The symptom is answers getting worse the longer a
conversation runs, which is easy to misdiagnose as a retrieval problem. A
`deque` with `maxlen` costs nothing and removes the failure mode.
