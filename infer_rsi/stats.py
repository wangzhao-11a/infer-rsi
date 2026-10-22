"""Small-sample statistics for paired A/B experiments.

Service-level benchmarks on a shared NPU host are noisy: run-to-run drift of a
few percent is normal, and with only 3 repetitions a t-test is not trustworthy.
We therefore report:

  * median per-rep improvement (robust to one bad rep),
  * a percentile bootstrap CI over the paired deltas,
  * the sign test (how many reps the candidate won),
  * the baseline coefficient of variation as an explicit noise gate.

The numbers are intentionally conservative: a candidate must clear both the
pre-registered minimum improvement and the lower CI bound.
"""
from __future__ import annotations

import math
import random
import statistics


def mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.fmean(values) if values else None


def median(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.median(values) if values else None


def stdev(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.stdev(values)


def cv_pct(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    avg = statistics.fmean(values)
    if avg == 0:
        return 0.0
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return abs(sd / avg) * 100.0


def improvement_pct(baseline: float, candidate: float, direction: str = "maximize") -> float:
    """Relative improvement in percent, positive means the candidate is better."""
    if baseline is None or candidate is None:
        return float("nan")
    if direction == "minimize":
        if candidate == 0:
            return float("inf") if baseline > 0 else 0.0
        return (baseline / candidate - 1.0) * 100.0
    if baseline == 0:
        return float("inf") if candidate > 0 else 0.0
    return (candidate / baseline - 1.0) * 100.0


def bootstrap_ci(values, confidence: float = 0.90, n_resamples: int = 2000, seed: int = 0):
    """Percentile bootstrap CI for the median.

    With n=3 the interval is wide on purpose; that is the honest answer.
    """
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return (None, None)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    stats = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistics.median(sample))
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = stats[int(alpha * (len(stats) - 1))]
    hi = stats[int((1.0 - alpha) * (len(stats) - 1))]
    return (lo, hi)


def paired_summary(baseline_values, candidate_values, direction: str = "maximize",
                   confidence: float = 0.90, seed: int = 0) -> dict:
    pairs = [
        (float(b), float(c))
        for b, c in zip(baseline_values, candidate_values)
        if b is not None and c is not None
    ]
    base = [p[0] for p in pairs]
    cand = [p[1] for p in pairs]
    deltas = [improvement_pct(b, c, direction) for b, c in pairs]
    finite = [d for d in deltas if math.isfinite(d)]
    ci_low, ci_high = bootstrap_ci(finite, confidence=confidence, seed=seed) if finite else (None, None)
    wins = sum(1 for d in deltas if math.isfinite(d) and d > 0)
    ties = sum(1 for d in deltas if math.isfinite(d) and d == 0)
    return {
        "n": len(pairs),
        "baseline_median": median(base),
        "candidate_median": median(cand),
        "baseline_mean": mean(base),
        "candidate_mean": mean(cand),
        "improvement_pct": median(finite) if finite else None,
        "improvement_mean_pct": mean(finite),
        "ci_low_pct": ci_low,
        "ci_high_pct": ci_high,
        "confidence": confidence,
        "wins": wins,
        "ties": ties,
        "losses": len(finite) - wins - ties,
        "baseline_cv_pct": cv_pct(base),
        "candidate_cv_pct": cv_pct(cand),
        "per_rep": [
            {
                "index": i,
                "baseline": pairs[i][0],
                "candidate": pairs[i][1],
                "improvement_pct": deltas[i],
            }
            for i in range(len(pairs))
        ],
    }


def required_wins(n: int) -> int:
    """Majority of paired reps, at least 1.  For n=3 -> 2."""
    return max(1, math.ceil(n * 0.6))