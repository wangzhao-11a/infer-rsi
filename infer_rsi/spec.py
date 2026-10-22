"""Scenario specification: the unit of work for one inference-tuning experiment.

A scenario pins down five things before any GPU time is spent:

    model x workload x hardware/stack x baseline x candidate

and the decision rule:

    primary metric + hard constraints + minimum improvement + noise threshold

The spec is deliberately declarative.  An agent may *propose* a new spec or new
overrides, but the harness only executes what the spec authorizes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

VALID_DIRECTIONS = ("maximize", "minimize")
VALID_OPS = ("<=", ">=", "<", ">", "==", "!=")
RESULT_PLACEHOLDER = "{result}"


class SpecError(ValueError):
    """Raised when a scenario file cannot be executed safely."""


@dataclass
class MetricSpec:
    key: str
    unit: str = ""
    direction: str = "maximize"

    @classmethod
    def from_dict(cls, value: Any) -> "MetricSpec":
        if isinstance(value, str):
            value = {"key": value}
        if not isinstance(value, dict) or not value.get("key"):
            raise SpecError("primary_metric must be a string or an object with a non-empty 'key'")
        direction = str(value.get("direction") or "maximize").lower()
        if direction not in VALID_DIRECTIONS:
            raise SpecError("primary_metric.direction must be one of %s" % (VALID_DIRECTIONS,))
        return cls(key=str(value["key"]), unit=str(value.get("unit") or ""), direction=direction)


@dataclass
class Constraint:
    key: str
    op: str
    value: float

    @classmethod
    def from_dict(cls, value: dict) -> "Constraint":
        if not isinstance(value, dict):
            raise SpecError("constraint entries must be objects")
        key = str(value.get("key") or "").strip()
        op = str(value.get("op") or "").strip()
        if not key:
            raise SpecError("constraint.key is required")
        if op not in VALID_OPS:
            raise SpecError("constraint.op must be one of %s, got %r" % (VALID_OPS, op))
        try:
            limit = float(value["value"])
        except (KeyError, TypeError, ValueError):
            raise SpecError("constraint %r needs a numeric 'value'" % key)
        return cls(key=key, op=op, value=limit)

    def check(self, actual: float) -> bool:
        if self.op == "<=":
            return actual <= self.value
        if self.op == ">=":
            return actual >= self.value
        if self.op == "<":
            return actual < self.value
        if self.op == ">":
            return actual > self.value
        if self.op == "==":
            return actual == self.value
        if self.op == "!=":
            return actual != self.value
        raise SpecError("unsupported op %r" % self.op)


@dataclass
class Scenario:
    id: str
    description: str
    repo: str
    backend: str
    hardware: dict
    stack: dict
    model: str
    topology: str
    workload: dict
    current_config: dict
    primary_metric: MetricSpec
    constraints: list
    baseline_command: Any
    candidate_command: Any
    quality_command: Any = None
    repetitions: int = 3
    min_improvement_pct: float = 3.0
    noise_cv_pct: float = 8.0
    max_minutes: float = 120.0
    workdir: str = "."
    env: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    path: str = ""

    # ------------------------------------------------------------------ helpers
    @property
    def hardware_key(self) -> str:
        model = str(self.hardware.get("device_model") or self.hardware.get("soc") or "unknown")
        ids = self.hardware.get("device_ids") or []
        host = str(self.hardware.get("host") or "local")
        return "%s-%s-%s" % (host, model, "-".join(str(x) for x in ids) or "all")

    @property
    def is_template(self) -> bool:
        return bool(self.metadata.get("template"))

    def candidate_config(self, overrides: dict | None = None) -> dict:
        cfg = dict(self.current_config or {})
        cfg.update(overrides or {})
        return cfg

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "repo": self.repo,
            "backend": self.backend,
            "hardware": self.hardware,
            "stack": self.stack,
            "model": self.model,
            "topology": self.topology,
            "workload": self.workload,
            "current_config": self.current_config,
            "primary_metric": {
                "key": self.primary_metric.key,
                "unit": self.primary_metric.unit,
                "direction": self.primary_metric.direction,
            },
            "constraints": [{"key": c.key, "op": c.op, "value": c.value} for c in self.constraints],
            "baseline_command": self.baseline_command,
            "candidate_command": self.candidate_command,
            "quality_command": self.quality_command,
            "repetitions": self.repetitions,
            "min_improvement_pct": self.min_improvement_pct,
            "noise_cv_pct": self.noise_cv_pct,
            "max_minutes": self.max_minutes,
            "workdir": self.workdir,
            "env": self.env,
            "metadata": self.metadata,
            "path": self.path,
        }


def _command_text(command: Any) -> str:
    if isinstance(command, (list, tuple)):
        return " ".join(str(x) for x in command)
    return str(command or "")


def _resolve_path(value: str, base_dir: str) -> str:
    value = os.path.expandvars(os.path.expanduser(str(value or ".")))
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def scenario_from_dict(data: dict, path: str = "") -> Scenario:
    if not isinstance(data, dict):
        raise SpecError("scenario must be a JSON object")

    for key in ("id", "baseline_command", "candidate_command", "primary_metric"):
        if not data.get(key):
            raise SpecError("scenario.%s is required" % key)

    for key in ("baseline_command", "candidate_command"):
        if RESULT_PLACEHOLDER not in _command_text(data[key]):
            raise SpecError("%s must contain the %s placeholder" % (key, RESULT_PLACEHOLDER))

    try:
        repetitions = int(data.get("repetitions", 3))
    except (TypeError, ValueError):
        raise SpecError("repetitions must be an integer")
    if repetitions < 1:
        raise SpecError("repetitions must be >= 1")
    if repetitions > 20:
        raise SpecError("repetitions > 20 is not a sensible default; raise it deliberately in code")

    base_dir = os.path.dirname(os.path.abspath(path)) if path else os.getcwd()
    workdir = _resolve_path(data.get("workdir") or ".", base_dir)
    repo = _resolve_path(data.get("repo") or ".", base_dir) if data.get("repo") else workdir

    scenario = Scenario(
        id=str(data["id"]),
        description=str(data.get("description") or ""),
        repo=repo,
        backend=str(data.get("backend") or "ascend"),
        hardware=dict(data.get("hardware") or {}),
        stack=dict(data.get("stack") or {}),
        model=str(data.get("model") or ""),
        topology=str(data.get("topology") or ""),
        workload=dict(data.get("workload") or {}),
        current_config=dict(data.get("current_config") or {}),
        primary_metric=MetricSpec.from_dict(data["primary_metric"]),
        constraints=[Constraint.from_dict(c) for c in (data.get("constraints") or [])],
        baseline_command=data["baseline_command"],
        candidate_command=data["candidate_command"],
        quality_command=data.get("quality_command"),
        repetitions=repetitions,
        min_improvement_pct=float(data.get("min_improvement_pct", 3.0)),
        noise_cv_pct=float(data.get("noise_cv_pct", 8.0)),
        max_minutes=float(data.get("max_minutes", 120.0)),
        workdir=workdir,
        env=dict(data.get("env") or {}),
        metadata=dict(data.get("metadata") or {}),
        path=os.path.abspath(path) if path else "",
    )

    if scenario.max_minutes <= 0:
        raise SpecError("max_minutes must be > 0")
    if scenario.min_improvement_pct < 0:
        raise SpecError("min_improvement_pct must be >= 0")
    if scenario.noise_cv_pct < 0:
        raise SpecError("noise_cv_pct must be >= 0")
    if not os.path.isdir(scenario.workdir):
        raise SpecError("workdir does not exist: %s" % scenario.workdir)

    return scenario


def load_scenario(path: str) -> Scenario:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        raise SpecError("cannot read scenario %s: %s" % (path, exc))
    except ValueError as exc:
        raise SpecError("scenario %s is not valid JSON: %s" % (path, exc))
    return scenario_from_dict(data, path)