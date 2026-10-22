"""Synthetic arm for the InferTune-RSI demo.  No NPU, no server, just metrics.

It exercises the real harness contract: read INFER_RSI_CONFIG, write a JSON
result to --result, vary with the knob so `sweep` shows a real curve.

    python demo/fake_server.py --mode baseline  --result out.json
    python demo/fake_server.py --mode candidate --result out.json --base-gain-pct 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "candidate"], default="baseline")
    parser.add_argument("--result", required=True)
    parser.add_argument("--base-gain-pct", type=float, default=None,
                        help="synthetic improvement before knob effects (candidate default 8)")
    parser.add_argument("--noise-pct", type=float, default=2.0)
    parser.add_argument("--noise-pattern", default="",
                        help="comma-separated per-rep noise percentages (overrides --noise-pct)")
    parser.add_argument("--ttft", type=float, default=900.0)
    parser.add_argument("--tpot", type=float, default=30.0)
    parser.add_argument("--memory", type=float, default=60.0)
    parser.add_argument("--output-len", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()

    if args.sleep > 0:
        time.sleep(args.sleep)
    if args.fail:
        print("simulated arm failure", file=sys.stderr)
        return 1

    config = {}
    raw = os.environ.get("INFER_RSI_CONFIG")
    if raw:
        try:
            config = json.loads(raw)
        except ValueError:
            config = {}

    if args.mode == "candidate":
        gain = 8.0 if args.base_gain_pct is None else float(args.base_gain_pct)
    else:
        gain = 0.0

    knob_gain = 0.0
    seqs = config.get("max_num_seqs")
    if args.mode == "candidate" and isinstance(seqs, (int, float)) and seqs > 72:
        # saturating curve: +2.9% at 90, +4.8% at 108
        knob_gain = 8.0 * (1.0 - math.exp(-(float(seqs) - 72.0) / 40.0))
    gain += knob_gain

    seed = args.seed if args.seed >= 0 else (time.time_ns() ^ (os.getpid() << 16))
    if args.noise_pattern:
        pattern = [float(x) for x in args.noise_pattern.split(",") if x.strip()]
        rep = int(os.environ.get("INFER_RSI_REP", "0") or 0)
        noise = pattern[rep % len(pattern)] if pattern else 0.0
    else:
        noise = random.Random(seed).gauss(0.0, max(0.0, args.noise_pct))

    throughput = 1000.0 * (1.0 + (gain + noise) / 100.0)
    ttft = args.ttft * (1.0 - 0.30 * gain / 100.0) * (1.0 + noise / 200.0)
    tpot = args.tpot * (1.0 + 0.20 * max(0.0, gain) / 100.0)
    e2e = ttft + tpot * args.output_len
    memory = args.memory + 0.05 * max(0.0, (seqs or 72) - 72)

    payload = {
        "metrics": {
            "output_throughput": round(throughput, 3),
            "ttft_p95_ms": round(ttft, 3),
            "tpot_p95_ms": round(tpot, 3),
            "e2e_p95_ms": round(e2e, 3),
            "memory_gb": round(memory, 3),
        },
        "meta": {
            "mode": args.mode,
            "config": config,
            "seed": seed,
            "gain_pct": round(gain, 4),
            "noise_pct": round(noise, 4),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.result)), exist_ok=True)
    with open(args.result, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("wrote %s throughput=%.1f" % (args.result, throughput))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())