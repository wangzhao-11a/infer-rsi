"""A/B harness: execute baseline vs candidate under one budget, with locks and gates.

Contract for an arm command:
    <command> ... {result}
It must start whatever it needs, run the workload, and write a JSON file at
{result} that contains either {"metrics": {...}} or flat numeric metrics.  The
harness never manages vLLM/PD servers itself; the launch/stop/health logic stays
in your own deployment and benchmark scripts.

Repetition order is ABBA: even reps run baseline first, odd reps candidate
first.  This cancels host drift over a long experiment.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from datetime import datetime

from . import gates
from .knowledge import DEFAULT_KNOWLEDGE_ROOT, KnowledgeStore
from .ledger import DEFAULT_LEDGER_ROOT, RunLedger, local_stamp, utc_now
from .report import render_report


class BlockedError(RuntimeError):
    def __init__(self, blockers):
        self.blockers = blockers
        super().__init__("scenario blocked by %d feasibility constraint(s)" % len(blockers))


class LockError(RuntimeError):
    pass


# --------------------------------------------------------------------------- utils
def _subst(command, mapping: dict):
    if isinstance(command, (list, tuple)):
        return [_subst_item(str(x), mapping) for x in command]
    return _subst_item(str(command), mapping)


def _subst_item(text: str, mapping: dict) -> str:
    for key, value in mapping.items():
        text = text.replace("{%s}" % key, str(value))
    return text


def _split_command(command):
    if isinstance(command, (list, tuple)):
        return [str(x) for x in command]
    return shlex.split(str(command), posix=True)


def _kill_tree(process) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, timeout=20)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _run_process(args, cwd, env, timeout_s, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    started = time.time()
    timed_out = False
    returncode = None
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        try:
            process = subprocess.Popen(
                args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                text=True, start_new_session=(os.name != "nt"),
            )
        except OSError as exc:
            log.write("failed to start: %s\n" % exc)
            return 127, False, 0.0
        try:
            returncode = process.wait(timeout=max(1.0, float(timeout_s)))
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(process)
            try:
                process.wait(timeout=10)
            except Exception:
                pass
            returncode = -9
    return returncode, timed_out, round(time.time() - started, 3)


def _flatten(payload: dict, prefix: str = "") -> dict:
    flat = {}
    if not isinstance(payload, dict):
        return flat
    for key, value in payload.items():
        name = "%s.%s" % (prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif isinstance(value, bool):
            flat[name] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


def load_result(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None, {}
    if not isinstance(data, dict):
        return data, {}
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else None
    flat = _flatten(metrics if metrics is not None else data)
    if metrics is not None:
        flat.update({k: v for k, v in _flatten(data).items() if k not in flat})
    return data, flat


def command_program(command) -> str:
    try:
        args = _split_command(command)
    except ValueError:
        return ""
    return args[0] if args else ""


def command_available(command, workdir: str) -> tuple[bool, str]:
    program = command_program(command)
    if not program or "{" in program:
        return False, "no executable in command"
    if os.path.isabs(program):
        return os.path.isfile(program), program
    if os.path.isfile(os.path.join(workdir, program)):
        return True, os.path.join(workdir, program)
    found = shutil.which(program)
    return (bool(found), found or program)


# --------------------------------------------------------------------------- lock
class RunLock:
    def __init__(self, root: str, scenario):
        self.root = os.path.join(os.path.abspath(root), "locks")
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario.hardware_key)
        self.path = os.path.join(self.root, key + ".lock")
        self.acquired = False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            try:
                out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid],
                                     capture_output=True, text=True, timeout=10)
                return str(pid) in out.stdout
            except Exception:
                return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def acquire(self) -> "RunLock":
        os.makedirs(self.root, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    with open(self.path, "r", encoding="utf-8") as fh:
                        held = json.load(fh)
                except (OSError, ValueError):
                    held = {}
                if not self._pid_alive(int(held.get("pid") or 0)):
                    try:
                        os.remove(self.path)
                    except OSError:
                        pass
                    continue
                raise LockError(
                    "device set %s is locked by pid %s (run %s). "
                    "Refusing to start a second experiment on the same cards."
                    % (self.path, held.get("pid"), held.get("run_id")))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "created": utc_now()}, fh)
            self.acquired = True
            return self
        raise LockError("could not acquire lock %s" % self.path)

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            os.remove(self.path)
        except OSError:
            pass
        self.acquired = False


# --------------------------------------------------------------------------- one arm
def _arm_env(scenario, config: dict, overrides: dict, run_dir: str,
             index: int, arm: str) -> dict:
    env = dict(os.environ)
    for key, value in (scenario.env or {}).items():
        env[str(key)] = str(value)
    env.update({
        "INFER_RSI_CONFIG": json.dumps(config, ensure_ascii=False),
        "INFER_RSI_BASELINE_CONFIG": json.dumps(scenario.current_config or {}, ensure_ascii=False),
        "INFER_RSI_OVERRIDES": json.dumps(overrides or {}, ensure_ascii=False),
        "INFER_RSI_RUN_DIR": run_dir,
        "INFER_RSI_SCENARIO": scenario.id,
        "INFER_RSI_REP": str(index),
        "INFER_RSI_ARM": arm,
    })
    return env


def _posix_path(path) -> str:
    """Keep Windows paths intact through shlex (which treats \\ as an escape)."""
    return str(path).replace("\\", "/")


def _run_arm(scenario, arm: str, index: int, run_dir: str, config: dict,
             overrides: dict, deadline: float, quiet: bool) -> dict:
    reps_dir = os.path.join(run_dir, "reps")
    result_path = os.path.join(reps_dir, "rep-%02d-%s.json" % (index, arm))
    log_path = os.path.join(run_dir, "logs", "rep-%02d-%s.log" % (index, arm))
    template = scenario.baseline_command if arm == "baseline" else scenario.candidate_command
    mapping = {
        "result": _posix_path(result_path),
        "run_dir": _posix_path(run_dir),
        "rep": index,
        "arm": arm,
        "scenario": scenario.id,
    }
    args = _split_command(_subst(template, mapping))
    remaining = deadline - time.time()
    if remaining <= 1.0:
        return {
            "index": index, "arm": arm, "returncode": None, "timed_out": False,
            "duration_s": 0.0, "result_path": result_path, "log_path": log_path,
            "metrics": None, "error": "budget exhausted before this arm", "args": args,
        }

    if not quiet:
        print("[infer-rsi] rep %d %-9s %s" % (index, arm, " ".join(args)), flush=True)
    returncode, timed_out, duration = _run_process(
        args, scenario.workdir, _arm_env(scenario, config, overrides, run_dir, index, arm),
        remaining, log_path,
    )
    data, metrics = load_result(result_path)
    error = None
    if timed_out:
        error = "timeout after %.1fs" % duration
    elif returncode != 0:
        error = "exit code %s" % returncode
    elif not metrics:
        error = "result JSON missing or has no numeric metrics"
    record = {
        "index": index, "arm": arm, "returncode": returncode, "timed_out": timed_out,
        "duration_s": duration, "result_path": result_path, "log_path": log_path,
        "metrics": metrics or None, "error": error, "args": args,
    }
    return record


# --------------------------------------------------------------------------- run
def run_scenario(scenario, overrides: dict | None = None, ledger_root: str = None,
                 knowledge_root: str = None, dry_run: bool = False, distill: bool = False,
                 allow_blocked: bool = False, tag: str = "", quiet: bool = False,
                 repetitions: int | None = None) -> dict:
    overrides = dict(overrides or {})
    ledger_root = ledger_root or DEFAULT_LEDGER_ROOT
    knowledge_root = knowledge_root or DEFAULT_KNOWLEDGE_ROOT
    ledger = RunLedger(ledger_root)
    store = KnowledgeStore(knowledge_root)

    blockers, warnings = store.check_feasibility(scenario)
    if blockers and not allow_blocked:
        raise BlockedError(blockers)

    reps = int(repetitions or scenario.repetitions)
    if reps < 1:
        raise ValueError("repetitions must be >= 1")
    candidate_config = scenario.candidate_config(overrides)

    if dry_run:
        mapping = {"result": "<result.json>", "run_dir": "<run_dir>", "scenario": scenario.id}
        return {
            "run_id": None,
            "scenario_id": scenario.id,
            "decision": "DRY_RUN",
            "reasons": ["dry run: commands were not executed"],
            "primary_metric": {"key": scenario.primary_metric.key,
                               "unit": scenario.primary_metric.unit,
                               "direction": scenario.primary_metric.direction},
            "stats": {},
            "gates": [],
            "constraints": [],
            "quality": [],
            "run_errors": [],
            "overrides": overrides,
            "candidate_config": candidate_config,
            "baseline_config": dict(scenario.current_config or {}),
            "repetitions": reps,
            "commands": {
                "baseline": _subst(scenario.baseline_command, mapping),
                "candidate": _subst(scenario.candidate_command, mapping),
                "quality": _subst(scenario.quality_command, mapping) if scenario.quality_command else None,
            },
            "warnings": [w.get("claim") for w in warnings],
            "started_at": utc_now(),
            "dry_run": True,
        }

    lock = RunLock(ledger_root, scenario).acquire()
    try:
        run_dir = ledger.create(scenario, candidate_config, overrides, tag=tag)
        run_id = os.path.basename(run_dir)
        started = time.time()
        deadline = started + scenario.max_minutes * 60.0

        baseline_records, candidate_records = [], []
        baseline_metrics, candidate_metrics = [], []
        quality_results = []
        run_errors = []

        for index in range(reps):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            arm_records = {}
            for arm in order:
                config = (scenario.current_config if arm == "baseline" else candidate_config) or {}
                record = _run_arm(scenario, arm, index, run_dir, config, overrides, deadline, quiet)
                arm_records[arm] = record
                meta_path = ledger.record_rep(run_dir, index, arm, {
                    k: v for k, v in record.items() if k != "metrics"
                })
                record["meta_path"] = meta_path
                if record.get("metrics"):
                    ledger.record_result(run_dir, index, arm, {
                        "metrics": record["metrics"],
                        "source": record["result_path"],
                    })
                if record.get("error"):
                    run_errors.append("rep %d %s: %s" % (index, arm, record["error"]))
                    if not quiet:
                        print("[infer-rsi]   -> %s" % record["error"], flush=True)

            baseline_records.append(arm_records.get("baseline"))
            candidate_records.append(arm_records.get("candidate"))
            baseline_metrics.append((arm_records.get("baseline") or {}).get("metrics"))
            candidate_metrics.append((arm_records.get("candidate") or {}).get("metrics"))

            # quality gate runs once per rep, after both arms
            if scenario.quality_command:
                baseline_path = (arm_records.get("baseline") or {}).get("result_path", "")
                candidate_path = (arm_records.get("candidate") or {}).get("result_path", "")
                if not (os.path.isfile(baseline_path) and os.path.isfile(candidate_path)):
                    quality_results.append(None)
                    run_errors.append("rep %d quality: missing arm result" % index)
                else:
                    q_mapping = {
                        "run_dir": _posix_path(run_dir), "rep": index, "scenario": scenario.id,
                        "baseline": _posix_path(baseline_path),
                        "candidate": _posix_path(candidate_path),
                        "result": _posix_path(candidate_path),
                    }
                    q_args = _split_command(_subst(scenario.quality_command, q_mapping))
                    q_log = os.path.join(run_dir, "logs", "rep-%02d-quality.log" % index)
                    if not quiet:
                        print("[infer-rsi] rep %d quality   %s" % (index, " ".join(q_args)), flush=True)
                    q_rc, q_timeout, q_duration = _run_process(
                        q_args, scenario.workdir,
                        _arm_env(scenario, candidate_config, overrides, run_dir, index, "quality"),
                        max(1.0, deadline - time.time()), q_log,
                    )
                    quality_results.append(q_rc == 0)
                    ledger.record_rep(run_dir, index, "quality", {
                        "index": index, "arm": "quality", "returncode": q_rc,
                        "timed_out": q_timeout, "duration_s": q_duration,
                        "log_path": q_log, "args": q_args,
                    })
                    if q_rc != 0 and not quiet:
                        print("[infer-rsi]   -> quality gate failed (exit %s)" % q_rc, flush=True)
            else:
                quality_results.append(None)

            failed = any((r or {}).get("error") for r in (arm_records.get("baseline"),
                                                          arm_records.get("candidate")))
            if failed or time.time() >= deadline:
                if not quiet:
                    print("[infer-rsi] stopping early (failure or budget)", flush=True)
                break

        result = gates.evaluate(
            scenario, baseline_metrics, candidate_metrics, quality_results, run_errors,
            expected_n=reps,
        )
        result.update({
            "run_id": run_id,
            "scenario_id": scenario.id,
            "scenario_path": scenario.path,
            "primary_metric": {
                "key": scenario.primary_metric.key,
                "unit": scenario.primary_metric.unit,
                "direction": scenario.primary_metric.direction,
            },
            "overrides": overrides,
            "candidate_config": candidate_config,
            "baseline_config": dict(scenario.current_config or {}),
            "repetitions": reps,
            "tag": tag,
            "started_at": utc_now(),
            "duration_s": round(time.time() - started, 3),
            "budget_minutes": scenario.max_minutes,
            "run_dir": run_dir,
            "warnings": [w.get("claim") for w in warnings],
            "dry_run": False,
        })

        if distill:
            card = store.distill(result, scenario)
            if card:
                result["knowledge_card"] = card.get("id")

        report = render_report(result, scenario)
        ledger.finish(run_dir, result, report)
        result["report_path"] = os.path.join(run_dir, "report.md")
        return result
    finally:
        lock.release()


# --------------------------------------------------------------------------- sweep
def sweep(scenario, knob: str, values: list, ledger_root: str = None,
          knowledge_root: str = None, repetitions: int | None = None,
          tag: str = "", quiet: bool = False) -> dict:
    rows = []
    for value in values:
        if not quiet:
            print("[infer-rsi] sweep %s=%r" % (knob, value), flush=True)
        summary = run_scenario(
            scenario, overrides={knob: value}, ledger_root=ledger_root,
            knowledge_root=knowledge_root, distill=False, tag=tag or "sweep",
            repetitions=repetitions, quiet=quiet,
        )
        rows.append({
            "value": value,
            "run_id": summary.get("run_id"),
            "decision": summary.get("decision"),
            "primary": (summary.get("stats") or {}).get("candidate_median"),
            "baseline": (summary.get("stats") or {}).get("baseline_median"),
            "improvement_pct": (summary.get("stats") or {}).get("improvement_pct"),
            "ci_low_pct": (summary.get("stats") or {}).get("ci_low_pct"),
            "ci_high_pct": (summary.get("stats") or {}).get("ci_high_pct"),
            "wins": (summary.get("stats") or {}).get("wins"),
            "n": (summary.get("stats") or {}).get("n"),
            "reasons": summary.get("reasons"),
            "run_dir": summary.get("run_dir"),
        })

    best = None
    for row in rows:
        if row["improvement_pct"] is None:
            continue
        if best is None or row["improvement_pct"] > best["improvement_pct"]:
            best = row
    passed = [row for row in rows if row["decision"] == "PASS"]
    top_pass = max(passed, key=lambda r: r["improvement_pct"]) if passed else None

    root = os.path.abspath(ledger_root or DEFAULT_LEDGER_ROOT)
    sweep_dir = os.path.join(root, "sweeps", "%s-%s-%s" % (
        local_stamp(), re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario.id),
        re.sub(r"[^A-Za-z0-9_.-]+", "_", knob)))
    os.makedirs(sweep_dir, exist_ok=True)
    result = {
        "scenario_id": scenario.id,
        "knob": knob,
        "values": values,
        "rows": rows,
        "best": best,
        "best_pass": top_pass,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    with open(os.path.join(sweep_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(sweep_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(_render_sweep(result))
    result["sweep_dir"] = sweep_dir
    return result


def _render_sweep(result: dict) -> str:
    lines = [
        "# Sweep: `%s` / `%s`" % (result["scenario_id"], result["knob"]),
        "",
        "| %s | baseline | candidate | Δ%% | 90%% CI | wins | decision |" % result["knob"],
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for row in result["rows"]:
        lines.append("| %s | %s | %s | %+.2f | %s..%s | %s/%s | %s |" % (
            row["value"],
            _fmt(row["baseline"]), _fmt(row["primary"]),
            row["improvement_pct"] if row["improvement_pct"] is not None else 0.0,
            _fmt(row["ci_low_pct"]), _fmt(row["ci_high_pct"]),
            row["wins"], row["n"], row["decision"],
        ))
    lines.append("")
    if result.get("best_pass"):
        lines.append("最佳且通过门禁：`%s`（%+.2f%%）。" % (
            result["best_pass"]["value"], result["best_pass"]["improvement_pct"]))
    elif result.get("best"):
        lines.append("没有候选通过门禁；最高观测为 `%s`（%+.2f%%），需增加重复或放宽预算后再判。"
                     % (result["best"]["value"], result["best"]["improvement_pct"]))
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "-"
    try:
        return "%.3f" % float(value)
    except (TypeError, ValueError):
        return str(value)