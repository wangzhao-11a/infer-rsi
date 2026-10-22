# Design notes

This document explains the parts that are easy to get wrong: what counts as a
result, how knowledge is scoped, and where the extension points are.

## 1. The decision pipeline

```
scenario.json
     │
     ├─ doctor   static checks: required fields, command placeholders,
     │           version pins, feasibility cards, device lock
     │
     ├─ plan     retrieve priors → rank allowed knob changes → hypotheses
     │
     ├─ run      for each rep: baseline → candidate → quality
     │           order alternates (ABBA), deadline enforced, process tree killed
     │
     ├─ gates    data → quality → constraints → noise → performance
     │
     ├─ ledger   append-only JSONL: decision, stats, config, paths, timestamps
     │
     └─ distill  PASS → strategy card
                 FAIL → negative card
                 INCONCLUSIVE → nothing
```

The ordering matters. Data completeness is checked before quality, quality
before constraints, and noise before performance. A candidate that is faster but
produces different output never reaches the performance gate.

## 2. Decision rules

PASS requires all of:

- every planned repetition produced a result;
- all quality checks passed;
- no hard constraint violated;
- baseline CV below the scenario's `noise_cv_pct`;
- `improvement_pct >= min_improvement_pct`;
- 90% bootstrap CI lower bound > 0;
- the candidate wins at least 60% of repetitions.

INCONCLUSIVE is returned when the baseline is too noisy, the CI crosses zero, or
there are fewer reps than planned. It is deliberately not treated as a weak PASS:
writing a card from an inconclusive run is how knowledge bases fill with noise.

## 3. Cards

A card is a JSON object:

```json
{
  "id": "kc-0002",
  "kind": "strategy",
  "claim": "raising max_num_seqs from 64 to 96 gains 4-6% throughput",
  "reason": "larger decode batches amortize weight reads",
  "scope": {"hardware": ["demo-accelerator"], "model": ["demo-model"],
            "topology": ["single"], "dtypes": ["bf16"]},
  "action": {"type": "config", "payload": {"max_num_seqs": 96}},
  "expected": {"metric_key": "output_throughput", "delta_pct": 4.0},
  "evidence": [{"run": "...", "measured_delta_pct": 4.8}],
  "stats": {"applied": 2, "success": 2, "fail": 0, "inconclusive": 0},
  "confidence": "verified",
  "invalidators": ["KV memory causes preemption", "TPOT exceeds SLO"]
}
```

Kinds:

| kind | meaning |
|---|---|
| `strategy` | a change that measured a gain under a scope |
| `negative` | a change that was rejected, with the reason |
| `feasibility` | a hard blocker/allowed verdict derived from code or docs |
| `runbook` | an operational procedure |
| `combo` | practices that co-occurred with a gain |

Scope matching is deterministic: a card whose hardware, model or stack does not
match the scenario is not retrieved. This is intentional — a wrong prior is
worse than no prior.

## 4. Promotion and narrowing

- A card distilled from a PASS starts at `candidate`.
- Replaying the same result in a clean process and passing again promotes it to
  `verified` and appends the second evidence entry.
- A verified card that fails on another device or version does not disappear.
  Its scope narrows (or an invalidator fires), so the next plan still knows that
  the approach exists but does not apply here.

## 5. Extension points

- **Backends.** `backend` is free-form. The demo backend uses synthetic servers;
  a real setup points at your launcher and benchmark scripts.
- **Metrics.** Any numeric key can be the primary metric (`direction` chooses
  maximize/minimize). Constraint keys are the same namespace.
- **Quality.** Any command that exits non-zero on a quality regression works.
  Token-level equality, logprob deltas, MMLU/GSM8K, WER and SSIM are all just
  commands.
- **Knowledge.** Cards are plain JSONL; the store can be pointed at a different
  directory with `--knowledge-root`.

## 6. Non-goals

- Multi-tenant scheduling, HA or cluster orchestration.
- Replacing a profiler. The tool records *whether* a change helped; the profiler
  explains *why*.
- Automatic knob search at scale. Sweep is a convenience for one knob at a time;
  a real search strategy belongs above this layer.

## 7. Failure modes the design guards against

| Failure | Guard |
|---|---|
| Comparing a noisy baseline | CV gate before any performance claim |
| Cherry-picking a good run | Pre-registered reps and bar; wins-per-rep check |
| Faster but wrong | Quality gate runs before performance |
| Remembering folklore | Cards require scope, evidence and invalidators |
| Repeating a known dead end | Negative and feasibility cards are retrieved too |
| Losing the audit trail | Append-only ledger with git SHA and raw paths |
| Secret leakage | Env allow-list; never print `.secrets`; reports contain config and metrics only |