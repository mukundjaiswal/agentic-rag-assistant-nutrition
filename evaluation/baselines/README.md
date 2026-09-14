# Baselines

Commit a report here once a configuration is accepted as the reference point:

```bash
agentic-rag evaluate --dataset evaluation/datasets/v1.jsonl \
  --out evaluation/baselines/v1.json
```

Later runs are then a diff against a committed number rather than an argument
from memory:

```bash
agentic-rag evaluate --dataset evaluation/datasets/v1.jsonl \
  --baseline evaluation/baselines/v1.json --tolerance 0.02
```

The command exits non-zero when a metric moves past the tolerance in the wrong
direction, which is what lets CI gate on it. Direction is per metric: higher
groundedness is better, more repairs per question is worse.

Re-baseline deliberately, in its own commit, with the reason in the message. A
baseline quietly refreshed alongside a behaviour change measures nothing.
