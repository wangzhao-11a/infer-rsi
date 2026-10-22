# infer-rsi

**An experiment ledger and statistical gate for inference tuning.**

Most tuning scripts produce numbers. This one produces *decisions* — and refuses
to record a decision it cannot defend.

Every experiment is a five-part contract: **model × workload × hardware/stack ×
baseline × candidate**. The result is one of three states:

- **PASS** — quality and hard constraints hold, the gain clears the pre-registered
  bar, the 90% CI lower bound is above zero, and the candidate wins at least 60%
  of repetitions.
- **FAIL** — quality or a constraint broke, the candidate is significantly worse,
  or the gain is below the bar.
- **INCONCLUSIVE** — baseline noise exceeds the threshold, the CI crosses zero,
  or there are too few repetitions. **No knowledge is written in this state.**

Three repetitions are a screening pass, not proof. A PASS can be replayed and
promoted to `verified` only after a second independent run.

---

## Why

Inference tuning accumulates folklore: "batch 96 was faster on that machine",
"prefix cache didn't help us". Six weeks later nobody remembers the model
version, the workload, or whether the difference was noise.

`infer-rsi` adds the missing layer between your launcher and your notes:

```
doctor ── spec / command / version / device lock
   │
plan ──── retrieve condition-matched priors (successes + failures)
   │
run ───── ABBA repetitions: baseline / candidate / quality
   │       deadline budget + device lock + process-tree cleanup
   │
gates ── data → quality → constraints → noise → performance
   │
ledger ─ immutable record: git SHA / dirty / env / config / raw results / logs
   │
distill ─ PASS → strategy card; FAIL → negative card; INCONCLUSIVE → no card
   │
next round ← priors re-ranked; conditions narrow with counter-examples
```

The unit of knowledge is a **card** carrying a `scope` (where it was measured)
and `evidence` (which run proved it). When a card fails on another device or
version, its scope narrows instead of the claim being silently kept.

---

## Quick start (no NPU or GPU required)

```bash
cd infer-rsi

# 1) preflight: spec, commands, versions, feasibility, device lock
python -m infer_rsi doctor --scenario scenarios/demo_pass.json

# 2) retrieve priors and generate candidate hypotheses
python -m infer_rsi plan --scenario scenarios/demo_pass.json

# 3) baseline vs candidate: 3 reps, ABBA, quality gate
python -m infer_rsi run --scenario scenarios/demo_pass.json --distill

# 4) report
python -m infer_rsi report --latest

# 5) sweep one knob
python -m infer_rsi sweep --scenario scenarios/demo_knob.json \
    --knob max_num_seqs --values 72,90,108

# 6) knowledge base
python -m infer_rsi knowledge search --scenario scenarios/demo_pass.json
python -m infer_rsi knowledge list
```

The demo scenarios use a synthetic backend (`demo/fake_server.py`), so the whole
state machine — including every failure branch — runs on a laptop.

| Scenario | Expected verdict | What it exercises |
|---|---|---|
| `demo_pass.json` | PASS | quality, constraints, noise, performance all pass |
| `demo_quality_fail.json` | FAIL | faster but wrong output must be rejected |
| `demo_constraint_fail.json` | FAIL | throughput up but TPOT violates the SLO |
| `demo_noisy.json` | INCONCLUSIVE | baseline CV above the threshold |
| `demo_worse.json` | FAIL | candidate significantly slower |
| `demo_blocked.json` | blocked at doctor | a feasibility card rejects the requested feature |
| `demo_knob.json` | sweep | only one knob value clears the bar |

---

## Scenario contract

A scenario is a JSON file that pins five things and nothing else:

```json
{
  "id": "demo-pass",
  "workdir": "..",
  "backend": "demo",
  "hardware": {"device_model": "demo-accelerator", "device_ids": [0]},
  "stack": {"vllm": "0.10.0"},
  "model": "demo-model",
  "workload": {"input_len": 8192, "output_len": 1000, "concurrency": 256},
  "current_config": {"max_num_seqs": 72},
  "primary_metric": {"key": "output_throughput", "unit": "tok/s", "direction": "maximize"},
  "constraints": [{"key": "tpot_p95_ms", "op": "<=", "value": 35}],
  "baseline_command":  "python bench.py --config baseline  --result {result}",
  "candidate_command": "python bench.py --config candidate --result {result}",
  "quality_command":   "python check_quality.py --baseline {baseline} --candidate {candidate}",
  "repetitions": 3,
  "min_improvement_pct": 3.0,
  "noise_cv_pct": 8.0
}
```

`{result}` is where the command must write its metrics. `{baseline}` and
`{candidate}` are filled in for the quality step. The tool never assumes how
your benchmark works — it only runs your commands and reads the numbers.

---

## Design principles

1. **Three states, not a pass rate.** "We couldn't tell" is a first-class result
   and writes no knowledge.
2. **Pre-registered thresholds.** The bar and noise threshold live in the
   scenario, not in the analysis, so they cannot be chosen after seeing the data.
3. **ABBA ordering.** Alternating baseline/candidate reduces drift bias.
4. **Scope is mandatory.** A card without a scope cannot be retrieved; hardware,
   model and stack mismatches exclude it.
5. **Negative cards are assets.** "This did not work, under these conditions,
   for this reason" prevents the same experiment from being repeated.
6. **The ledger is immutable.** Every run records git SHA, dirty state, env,
   config, raw results and logs.

---

## What this is not

- Not a benchmark suite. It wraps *your* commands.
- Not a profiler. Use `nsys`, `msprof` or `torch.profiler` for that.
- Not a scheduler or a deployment tool.
- Not a substitute for statistical power: with 3 reps, PASS is a screening
  decision, not a publication-grade result.

---

## Install

```bash
pip install -e .
python -m unittest discover -s tests -v
```

Pure standard library at runtime. Python 3.10+.

---

## Related

- [kernel-agent](https://github.com/wangzhao-11a/kernel-agent) — operator-level optimization loop with scoped strategy memory.
- [agent-harness](https://github.com/wangzhao-11a/agent-harness) — zero-dependency harness runtime for LLM agents.

## License

Apache-2.0 — see `LICENSE`.