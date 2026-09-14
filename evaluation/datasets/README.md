# Evaluation datasets

Datasets are **not** committed. The corpus these projects were built against is
not mine to redistribute.

`agentic-rag evaluate` expects JSON Lines, one object per line:

```json
{"question": "How is iron absorbed?"}
{"question": "What competes with iron?", "reference": "Calcium.", "slice": "interactions"}
```

| Field | Required | Purpose |
|---|---|---|
| `question` | yes | Put to the agent |
| `reference` | no | Gold answer, for reference-based scoring |
| any other key | no | Preserved as metadata; use it for slice labels |

## Version the filename

Name the file `v1.jsonl`, `v2.jsonl` and so on, and never edit a version in
place. Every report records the dataset it ran against, and two reports are
comparable only if that string matches. Editing `v1.jsonl` silently invalidates
every baseline committed against it.
