"""Synthetic quality gate for the demo.  Exit 0 = outputs acceptable."""
from __future__ import annotations

import argparse
import json
import sys


def _load(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()

    if args.fail:
        print("simulated quality regression (SSIM/PSNR or exact-match gate)", file=sys.stderr)
        return 1

    base = (_load(args.baseline).get("metrics") or {})
    cand = (_load(args.candidate).get("metrics") or {})
    if not base or not cand:
        print("missing result metrics", file=sys.stderr)
        return 1
    if cand.get("output_throughput", 0) <= 0:
        print("candidate produced no tokens", file=sys.stderr)
        return 1
    if cand.get("tpot_p95_ms", 0) > 1000:
        print("candidate decode is pathologically slow", file=sys.stderr)
        return 1
    print("quality ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())