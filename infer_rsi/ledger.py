"""Immutable experiment ledger.

Every run gets a directory containing:

    run.json                 scenario snapshot + git/env fingerprint + candidate config
    reps/rep-<i>-<arm>.json  raw result JSON produced by the arm
    reps/rep-<i>-<arm>.meta.json  exit code, duration, log path
    logs/rep-<i>-<arm>.log   combined stdout/stderr
    summary.json             gate result + statistics
    report.md                human-readable report

and one append-only row in runs/ledger.jsonl.  The ledger is the source of truth
for distillation: a knowledge card without a run id is not evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

DEFAULT_LEDGER_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _git(args: list, cwd: str) -> str:
    try:
        out = subprocess.run(
            ["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=20
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def git_fingerprint(repo: str) -> dict:
    if not repo or not os.path.isdir(repo):
        return {"available": False, "reason": "repo missing"}
    head = _git(["rev-parse", "HEAD"], repo)
    if not head:
        return {"available": False, "reason": "not a git worktree"}
    status = _git(["status", "--porcelain"], repo)
    diff = _git(["diff", "HEAD"], repo)
    return {
        "available": True,
        "head": head,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
        "dirty": bool(status),
        "status_sha": _sha256_text(status),
        "diff_sha": _sha256_text(diff),
        "status_lines": status.splitlines()[:50],
    }


def env_fingerprint() -> dict:
    whitelist = (
        "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "HCCL_BUFFSIZE",
        "HCCL_IF_IP", "VLLM_USE_V2_MODEL_RUNNER", "LD_PRELOAD", "OMP_NUM_THREADS",
        "VLLM_ASCEND_ENABLE_NZ", "TASK_QUEUE_ENABLE",
    )
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "env": {k: os.environ.get(k) for k in whitelist if os.environ.get(k) is not None},
    }


def _atomic_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


class RunLedger:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.runs_dir = self.root
        self.ledger_path = os.path.join(self.runs_dir, "ledger.jsonl")
        os.makedirs(self.runs_dir, exist_ok=True)

    # ------------------------------------------------------------------ create
    def create(self, scenario, candidate_config: dict, overrides: dict,
               tag: str = "") -> str:
        stamp = local_stamp()
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in scenario.id)[:40]
        short = _sha256_text(json.dumps({
            "scenario": scenario.to_dict(),
            "candidate_config": candidate_config,
        }, sort_keys=True, default=str))[:8]
        run_id = "%s-%s-%s" % (stamp, slug, short)
        run_dir = os.path.join(self.runs_dir, run_id)
        suffix = 1
        while os.path.exists(run_dir):
            suffix += 1
            run_id = "%s-%s-%s-%d" % (stamp, slug, short, suffix)
            run_dir = os.path.join(self.runs_dir, run_id)
        os.makedirs(os.path.join(run_dir, "reps"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

        meta = {
            "run_id": run_id,
            "tag": tag,
            "scenario": scenario.to_dict(),
            "scenario_path": scenario.path,
            "baseline_config": dict(scenario.current_config or {}),
            "candidate_config": candidate_config,
            "overrides": overrides,
            "git": git_fingerprint(scenario.repo),
            "env": env_fingerprint(),
            "started_at": utc_now(),
            "run_dir": run_dir,
        }
        _atomic_json(os.path.join(run_dir, "run.json"), meta)
        return run_dir

    # ------------------------------------------------------------------ reps
    def record_rep(self, run_dir: str, index: int, arm: str, payload: dict) -> str:
        path = os.path.join(run_dir, "reps", "rep-%02d-%s.meta.json" % (index, arm))
        _atomic_json(path, payload)
        return path

    def record_result(self, run_dir: str, index: int, arm: str, payload) -> str:
        path = os.path.join(run_dir, "reps", "rep-%02d-%s.json" % (index, arm))
        _atomic_json(path, payload)
        return path

    # ------------------------------------------------------------------ finish
    def finish(self, run_dir: str, summary: dict, report_md: str) -> None:
        summary = dict(summary)
        summary["finished_at"] = utc_now()
        _atomic_json(os.path.join(run_dir, "summary.json"), summary)
        with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
            fh.write(report_md)

        stats = summary.get("stats") or {}
        row = {
            "run_id": summary.get("run_id"),
            "scenario_id": summary.get("scenario_id"),
            "decision": summary.get("decision"),
            "primary_metric": (summary.get("primary_metric") or {}),
            "improvement_pct": stats.get("improvement_pct"),
            "ci_low_pct": stats.get("ci_low_pct"),
            "ci_high_pct": stats.get("ci_high_pct"),
            "wins": stats.get("wins"),
            "n": stats.get("n"),
            "baseline_cv_pct": stats.get("baseline_cv_pct"),
            "overrides": summary.get("overrides"),
            "candidate_config": summary.get("candidate_config"),
            "tag": summary.get("tag"),
            "finished_at": summary["finished_at"],
            "run_dir": run_dir,
        }
        _append_jsonl(self.ledger_path, row)

    # ------------------------------------------------------------------ query
    def list_runs(self) -> list:
        rows = []
        if not os.path.isfile(self.ledger_path):
            return rows
        with open(self.ledger_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows

    def latest(self, scenario_id: str = None) -> dict | None:
        rows = self.list_runs()
        if scenario_id:
            rows = [r for r in rows if r.get("scenario_id") == scenario_id]
        return rows[-1] if rows else None

    def find(self, run_id: str) -> str | None:
        path = os.path.join(self.runs_dir, run_id)
        if os.path.isdir(path):
            return path
        for row in self.list_runs():
            if row.get("run_id") == run_id:
                candidate = row.get("run_dir") or os.path.join(self.runs_dir, run_id)
                if os.path.isdir(candidate):
                    return candidate
        return None

    def read_summary(self, run_dir: str) -> dict:
        return read_json(os.path.join(run_dir, "summary.json"), {}) or {}

    def read_report(self, run_dir: str) -> str:
        path = os.path.join(run_dir, "report.md")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            return ""


def wait_for_file(path: str, timeout_s: float, poll_s: float = 0.1) -> bool:
    deadline = time.time() + max(0.0, timeout_s)
    while time.time() < deadline:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return True
        time.sleep(poll_s)
    return os.path.isfile(path) and os.path.getsize(path) > 0