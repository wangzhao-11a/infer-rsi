"""Decision gates: correctness first, then noise, then performance.

A candidate is only a "keep" when every gate passes.  The three-state result
mirrors practical experiment review:

    PASS          keep the change (quality + constraints + significant gain)
    FAIL          reject it (quality/constraint violation, worse, or below bar)
    INCONCLUSIVE  run more/better experiments (noisy or direction unclear)

The order matters: a fast wrong answer must never reach the performance gate.
"""
from __future__ import annotations

from . import stats as st


def _constraint_rows(scenario, candidate_metrics: list) -> tuple[list, bool]:
    rows = []
    ok = True
    for constraint in scenario.constraints:
        values = [m.get(constraint.key) for m in candidate_metrics]
        missing = [i for i, v in enumerate(values) if v is None]
        if missing:
            ok = False
            rows.append({
                "key": constraint.key,
                "op": constraint.op,
                "limit": constraint.value,
                "observed": None,
                "passed": False,
                "detail": "missing in rep(s) %s" % missing,
            })
            continue
        passed = all(constraint.check(float(v)) for v in values)
        ok = ok and passed
        rows.append({
            "key": constraint.key,
            "op": constraint.op,
            "limit": constraint.value,
            "observed": min(values) if constraint.op in ("<=", "<") else max(values),
            "observed_all": values,
            "passed": passed,
            "detail": "worst-case over %d rep(s)" % len(values),
        })
    return rows, ok


def _quality_rows(quality_results: list, has_quality_command: bool) -> tuple[list, bool]:
    if not has_quality_command:
        return ([{"rep": i, "passed": None, "detail": "not configured"} for i in range(len(quality_results))], True)
    rows = []
    ok = True
    for i, value in enumerate(quality_results):
        passed = bool(value)
        ok = ok and passed
        rows.append({
            "rep": i,
            "passed": passed,
            "detail": "quality command exit 0" if passed else
                      ("quality command failed" if value is not None else "quality command did not run"),
        })
        if value is None:
            ok = False
    return rows, ok


def evaluate(scenario, baseline_metrics: list, candidate_metrics: list,
             quality_results: list, run_errors: list | None = None,
             dry_run: bool = False, expected_n: int | None = None) -> dict:
    run_errors = list(run_errors or [])
    expected_n = int(expected_n or len(baseline_metrics) or 0)
    n = len(baseline_metrics)
    gates = []
    reasons = []

    if dry_run:
        return {
            "decision": "DRY_RUN",
            "reasons": ["dry run: commands were not executed"],
            "gates": [{"name": "dry_run", "status": "skipped", "detail": "no execution"}],
            "stats": {},
            "constraints": [],
            "quality": [],
            "run_errors": run_errors,
        }

    base_ok = [m for m in baseline_metrics if m is not None]
    cand_ok = [m for m in candidate_metrics if m is not None]

    # ---------------------------------------------------------------- data gate
    if not base_ok and not cand_ok:
        decision = "FAIL"
        reasons.append("both arms produced no parseable result")
        gates.append({"name": "data", "status": "fail", "detail": "no results"})
        return _result(decision, reasons, gates, {}, [], [], run_errors)

    if not cand_ok:
        decision = "FAIL"
        reasons.append("candidate produced no parseable result")
        gates.append({"name": "data", "status": "fail", "detail": "candidate missing"})
        return _result(decision, reasons, gates, {}, [], [], run_errors)

    if not base_ok:
        decision = "INCONCLUSIVE"
        reasons.append("baseline produced no parseable result; cannot compare")
        gates.append({"name": "data", "status": "inconclusive", "detail": "baseline missing"})
        return _result(decision, reasons, gates, {}, [], [], run_errors)

    if len(base_ok) < n or len(cand_ok) < n:
        decision = "INCONCLUSIVE"
        reasons.append("only %d/%d baseline and %d/%d candidate reps produced results"
                       % (len(base_ok), n, len(cand_ok), n))
        gates.append({"name": "data", "status": "inconclusive", "detail": "incomplete reps"})
        # still fall through to compute partial statistics

    complete = len(base_ok) >= expected_n and len(cand_ok) >= expected_n
    gates.append({
        "name": "reps",
        "status": "pass" if complete else "inconclusive",
        "detail": "%d/%d baseline and %d/%d candidate reps" % (
            len(base_ok), expected_n, len(cand_ok), expected_n),
    })
    if not complete:
        reasons.append("only %d/%d planned repetitions completed; do not decide on partial data"
                       % (min(len(base_ok), len(cand_ok)), expected_n))

    # ---------------------------------------------------------------- quality gate
    quality_rows, quality_ok = _quality_rows(quality_results, bool(scenario.quality_command))
    gates.append({
        "name": "quality",
        "status": "pass" if quality_ok else "fail",
        "detail": "all output gates passed" if quality_ok else "output quality gate failed",
    })
    if not quality_ok:
        reasons.append("quality gate failed: do not retain this candidate")

    # ------------------------------------------------------------ constraints gate
    constraint_rows, constraints_ok = _constraint_rows(scenario, cand_ok)
    gates.append({
        "name": "constraints",
        "status": "pass" if constraints_ok else "fail",
        "detail": "all hard constraints met" if constraints_ok else "at least one hard constraint violated",
    })
    if not constraints_ok:
        violated = [r["key"] for r in constraint_rows if not r["passed"]]
        reasons.append("hard constraint(s) violated: %s" % ", ".join(violated))

    # ---------------------------------------------------------------- statistics
    primary = scenario.primary_metric.key
    base_vals = [m.get(primary) for m in baseline_metrics]
    cand_vals = [m.get(primary) for m in candidate_metrics]
    summary = st.paired_summary(base_vals, cand_vals, scenario.primary_metric.direction)
    if summary["baseline_median"] is None or summary["candidate_median"] is None:
        decision = "INCONCLUSIVE"
        reasons.append("primary metric %r missing from result JSON" % primary)
        gates.append({"name": "metric", "status": "fail", "detail": "primary metric missing"})
        return _result(decision, reasons, gates, summary, constraint_rows, quality_rows, run_errors)

    # ---------------------------------------------------------------- noise gate
    cv = summary.get("baseline_cv_pct")
    noise_ok = cv is not None and cv <= scenario.noise_cv_pct
    gates.append({
        "name": "noise",
        "status": "pass" if noise_ok else "inconclusive",
        "detail": "baseline CV %.2f%% (limit %.2f%%)" % (cv or 0.0, scenario.noise_cv_pct),
    })

    # ------------------------------------------------------------ performance gate
    delta = summary.get("improvement_pct")
    ci_low = summary.get("ci_low_pct")
    ci_high = summary.get("ci_high_pct")
    wins = summary.get("wins", 0)
    need_wins = st.required_wins(expected_n)
    perf_status = "pass"
    perf_detail = "median %+.2f%% (90%% CI %+.2f%%..%+.2f%%), wins %d/%d" % (
        delta or 0.0, ci_low or 0.0, ci_high or 0.0, wins, summary.get("n", 0))

    if delta is not None and delta < 0 and ci_high is not None and ci_high < 0:
        perf_status = "fail"
        if quality_ok and constraints_ok:
            reasons.append("candidate is significantly worse (%+.2f%%)" % delta)
    elif ci_low is None or ci_low <= 0:
        perf_status = "inconclusive"
        if quality_ok and constraints_ok:
            reasons.append("improvement is not distinguishable from zero (CI low %s)" % ci_low)
    elif delta < scenario.min_improvement_pct:
        perf_status = "fail"
        if quality_ok and constraints_ok:
            reasons.append("improvement %+.2f%% is below the pre-registered %.2f%% bar"
                           % (delta, scenario.min_improvement_pct))
    elif wins < need_wins:
        perf_status = "inconclusive"
        if quality_ok and constraints_ok:
            reasons.append("candidate won only %d/%d paired reps (need %d)"
                           % (wins, summary.get("n", 0), need_wins))

    gates.append({"name": "performance", "status": perf_status, "detail": perf_detail})

    # ---------------------------------------------------------------- decision
    # correctness/quality and hard constraints are validity failures independent of noise;
    # everything else must first pass the noise and completeness gates before it can
    # be called a win or a regression.
    hard_fail = (not quality_ok) or (not constraints_ok)
    if hard_fail:
        decision = "FAIL"
    elif (not noise_ok) or (not complete):
        decision = "INCONCLUSIVE"
        if not noise_ok:
            reasons.append("baseline noise CV %.2f%% exceeds %.2f%%; repeat on a quieter host or add reps"
                           % (cv or 0.0, scenario.noise_cv_pct))
    elif perf_status == "fail":
        decision = "FAIL"
    elif perf_status == "inconclusive":
        decision = "INCONCLUSIVE"
    else:
        decision = "PASS"

    if decision == "PASS" and not reasons:
        reasons.append("quality, constraints, noise and performance gates all passed")

    result = _result(decision, reasons, gates, summary, constraint_rows, quality_rows, run_errors)
    result["constraints_ok"] = constraints_ok
    result["quality_ok"] = quality_ok
    result["noise_ok"] = noise_ok
    result["need_wins"] = need_wins
    result["expected_n"] = expected_n
    return result


def _result(decision, reasons, gates, summary, constraints, quality, run_errors):
    return {
        "decision": decision,
        "reasons": list(dict.fromkeys(reasons)),
        "gates": gates,
        "stats": summary,
        "constraints": constraints,
        "quality": quality,
        "run_errors": list(run_errors),
    }