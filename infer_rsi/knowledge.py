"""Knowledge cards: successes, failures and hard feasibility constraints.

The card is the unit of accumulated experience.  Unlike a "best practices" doc,
every card carries a *scope* (where it was measured) and *evidence* (which run
proved it), so the next round can retrieve it by condition match and can narrow
it when a counter-example appears on another device/version.

Kinds:
    strategy     a change that measured a gain under a scope
    negative     a change that was rejected under a scope (with the reason)
    feasibility  a hard blocker/allowed verdict derived from code/docs
    runbook      an operational procedure (how to bring up/validate)
    combo        practices that co-occurred with a gain
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone

DEFAULT_LEDGER_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
DEFAULT_KNOWLEDGE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(value) -> str:
    return " ".join(str(value or "").lower().split())


def _contains_any(value: str, candidates) -> bool:
    value = _norm(value)
    if not value:
        return False
    for candidate in candidates or []:
        cand = _norm(candidate)
        if cand and (cand in value or value in cand):
            return True
    return False


def _version_tuple(text: str) -> tuple:
    nums = re.findall(r"\d+", str(text or ""))
    return tuple(int(n) for n in nums[:5])


def version_ok(actual: str, wanted: str) -> bool:
    """Tolerant version check for values like '0.30.0rc2' and '>=0.27.1rc1'."""
    wanted = str(wanted or "").strip()
    if wanted in ("", "*", "any"):
        return True
    match = re.match(r"^(>=|<=|>|<|==|=)?\s*(.+)$", wanted)
    op = match.group(1) or "=="
    target = match.group(2)
    a = _version_tuple(actual)
    t = _version_tuple(target)
    if not a or not t:
        return _norm(actual) == _norm(target)
    width = max(len(a), len(t))
    a = a + (0,) * (width - len(a))
    t = t + (0,) * (width - len(t))
    if op in ("=", "=="):
        return a == t
    if op == ">=":
        return a >= t
    if op == "<=":
        return a <= t
    if op == ">":
        return a > t
    if op == "<":
        return a < t
    return False


def _empty_stats() -> dict:
    return {
        "applied": 0,
        "success": 0,
        "fail": 0,
        "inconclusive": 0,
        "ema_delta": None,
        "calibration_mae": None,
    }


def _blank_card() -> dict:
    return {
        "id": "",
        "key": "",
        "kind": "strategy",
        "claim": "",
        "scope": {},
        "action": {},
        "expected": {},
        "evidence": [],
        "stats": _empty_stats(),
        "confidence": "candidate",
        "invalidators": [],
        "tags": [],
        "source": "",
        "created": utc_now(),
        "updated": utc_now(),
    }


def _hardware_matches(scope: dict, scenario) -> bool:
    wanted = scope.get("hardware") or []
    if not wanted:
        return False
    actual = " ".join(str(x) for x in (
        scenario.hardware.get("device_model"),
        scenario.hardware.get("soc"),
        scenario.hardware.get("host"),
    ) if x)
    return _contains_any(actual, wanted)


def scope_match(scope: dict, scenario) -> float:
    """Return 0..1; mismatches pull the score down but do not hard-filter."""
    parts = []

    if scope.get("hardware"):
        parts.append((0.35, 1.0 if _hardware_matches(scope, scenario) else -0.9))
    if scope.get("model"):
        parts.append((0.25, 1.0 if _contains_any(scenario.model, scope["model"]) else -0.8))
    if scope.get("topology"):
        parts.append((0.15, 1.0 if _contains_any(scenario.topology, scope["topology"]) else -0.7))

    for key, wanted in (scope.get("stack") or {}).items():
        actual = scenario.stack.get(key)
        if actual is None:
            parts.append((0.12, -0.4))
        else:
            parts.append((0.15, 1.0 if version_ok(str(actual), str(wanted)) else -0.9))

    workload = scope.get("workload") or {}
    for key in ("input_len", "output_len", "concurrency"):
        if key not in workload:
            continue
        actual = scenario.workload.get(key)
        if not actual:
            parts.append((0.05, -0.2))
            continue
        ratio = max(float(actual), float(workload[key])) / max(1.0, min(float(actual), float(workload[key])))
        parts.append((0.06, 1.0 if ratio <= 2.0 else -0.5))

    dtypes = scope.get("dtypes") or []
    if dtypes:
        actual = str(scenario.metadata.get("dtype") or scenario.stack.get("dtype") or "")
        parts.append((0.10, 1.0 if _contains_any(actual, dtypes) else -0.6))

    tags = scope.get("tags") or []
    if tags:
        actual = scenario.metadata.get("tags") or []
        overlap = len({_norm(t) for t in tags} & {_norm(t) for t in actual})
        parts.append((0.08, min(1.0, overlap / max(1, len(tags)))))

    if not parts:
        return 0.5
    total = sum(w for w, _ in parts)
    raw = sum(w * s for w, s in parts) / total
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def _quality(card: dict) -> float:
    stats = card.get("stats") or {}
    applied = int(stats.get("applied") or 0)
    success = int(stats.get("success") or 0)
    rate = (success + 1.0) / (applied + 2.0)  # Laplace smoothing
    ema = stats.get("ema_delta")
    ema_score = 0.5 if ema is None else 0.5 + max(-0.5, min(0.5, float(ema) / 20.0))
    mae = stats.get("calibration_mae")
    calibration = 1.0 if mae is None else max(0.0, 1.0 - float(mae) / 20.0)
    return 0.5 * rate + 0.3 * ema_score + 0.2 * calibration


def _confidence(card: dict) -> float:
    return {
        "verified": 0.85,
        "candidate": 0.45,
        "seed": 0.22,
        "refuted": 0.05,
    }.get(str(card.get("confidence") or "candidate"), 0.3)


def _explore(card: dict) -> float:
    applied = int((card.get("stats") or {}).get("applied") or 0)
    return 1.0 / math.sqrt(1.0 + applied)


class KnowledgeStore:
    def __init__(self, root: str = DEFAULT_KNOWLEDGE_ROOT):
        self.root = os.path.abspath(root)
        self.cards_path = os.path.join(self.root, "cards.jsonl")
        self.cards = {}
        self.load()

    # ------------------------------------------------------------------ persistence
    def load(self) -> None:
        if not os.path.isfile(self.cards_path):
            return
        with open(self.cards_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    card = json.loads(line)
                except ValueError:
                    continue
                if card.get("id"):
                    card.setdefault("stats", _empty_stats())
                    card["key"] = card.get("key") or self._card_key(card)
                    self.cards[card["id"]] = card

    def save(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        tmp = self.cards_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for card_id in sorted(self.cards):
                fh.write(json.dumps(self.cards[card_id], ensure_ascii=False) + "\n")
        os.replace(tmp, self.cards_path)

    def _next_id(self) -> str:
        top = 0
        for card_id in self.cards:
            match = re.match(r"^kc-(\d+)$", card_id)
            if match:
                top = max(top, int(match.group(1)))
        return "kc-%04d" % (top + 1)

    def _card_key(self, card: dict) -> str:
        payload = json.dumps({
            "kind": card.get("kind"),
            "action": card.get("action"),
            "scope": card.get("scope"),
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ write
    def add_card(self, card: dict) -> dict:
        record = _blank_card()
        record.update(card or {})
        record["scope"] = dict(record.get("scope") or {})
        record["action"] = dict(record.get("action") or {})
        record["expected"] = dict(record.get("expected") or {})
        record["stats"] = dict(_empty_stats(), **(record.get("stats") or {}))
        record["evidence"] = list(record.get("evidence") or [])
        record["tags"] = list(record.get("tags") or [])
        record["invalidators"] = list(record.get("invalidators") or [])

        if not record.get("claim"):
            raise ValueError("card.claim is required")
        record["key"] = record.get("key") or self._card_key(record)

        existing = None
        for cid, current in self.cards.items():
            if current.get("key") == record["key"] or (record.get("id") and cid == record["id"]):
                existing = current
                break

        if existing:
            for field in ("claim", "scope", "action", "expected", "source", "confidence"):
                if record.get(field):
                    existing[field] = record[field]
            existing["tags"] = sorted(set(existing.get("tags") or []) | set(record["tags"]))
            existing["invalidators"] = sorted(set(existing.get("invalidators") or []) | set(record["invalidators"]))
            existing["updated"] = utc_now()
            self.save()
            return existing

        record["id"] = record.get("id") or self._next_id()
        record["created"] = record.get("created") or utc_now()
        record["updated"] = utc_now()
        self.cards[record["id"]] = record
        self.save()
        return record

    # ------------------------------------------------------------------ retrieve
    def retrieve(self, scenario, query: str = "", top_k: int = 5,
                 min_match: float = 0.35, kinds=None) -> list:
        kinds = set(kinds or [])
        query_norm = _norm(query)
        scored = []
        for card in self.cards.values():
            if kinds and card.get("kind") not in kinds:
                continue
            if card.get("kind") == "feasibility" and card.get("verdict") == "blocked":
                # blockers are reported by check_feasibility, not offered as priors
                continue
            scope = card.get("scope") or {}
            # a card measured on a different model is not a prior for this model
            if scope.get("model") and not _contains_any(scenario.model, scope["model"]):
                continue
            match = scope_match(scope, scenario)
            if match < min_match:
                continue
            score = 0.45 * match + 0.30 * _quality(card) + 0.15 * _confidence(card) + 0.10 * _explore(card)
            if query_norm:
                haystack = " ".join([
                    _norm(card.get("claim")),
                    _norm(card.get("action")),
                    _norm(card.get("tags")),
                    _norm(card.get("source")),
                ])
                if query_norm in haystack:
                    score += 0.05
            scored.append((card, score, match))
        scored.sort(key=lambda item: -item[1])
        return [(card, round(score, 4), round(match, 4)) for card, score, match in scored[:top_k]]

    def check_feasibility(self, scenario) -> tuple[list, list]:
        """Return (blockers, warnings) for features the scenario requests."""
        features = {str(x) for x in (scenario.metadata.get("requested_features") or [])}
        blockers, warnings = [], []
        for card in self.cards.values():
            if card.get("kind") != "feasibility":
                continue
            scope = card.get("scope") or {}
            if not _hardware_matches(scope, scenario):
                continue
            if scope.get("model") and not _contains_any(scenario.model, scope["model"]):
                continue
            # if the card pins a stack version and the scenario does not match, skip it
            stack_ok = True
            for key, wanted in (scope.get("stack") or {}).items():
                actual = scenario.stack.get(key)
                if actual is not None and not version_ok(str(actual), str(wanted)):
                    stack_ok = False
                    break
            if not stack_ok:
                continue
            affected = set(card.get("blocks_features") or []) & features
            if not affected:
                continue
            if card.get("verdict") == "blocked":
                blockers.append(card)
            else:
                warnings.append(card)
        return blockers, warnings

    def search(self, query: str, top_k: int = 10) -> list:
        q = _norm(query)
        rows = []
        for card in self.cards.values():
            haystack = " ".join([
                _norm(card.get("id")), _norm(card.get("claim")), _norm(card.get("kind")),
                _norm(card.get("tags")), _norm(card.get("scope")), _norm(card.get("action")),
                _norm(card.get("source")),
            ])
            if q and q not in haystack:
                continue
            rows.append((card, _quality(card) + 0.2 * _confidence(card)))
        rows.sort(key=lambda item: -item[1])
        return [card for card, _ in rows[:top_k]]

    # ------------------------------------------------------------------ distill
    def distill(self, summary: dict, scenario) -> dict | None:
        decision = summary.get("decision")
        if decision not in ("PASS", "FAIL"):
            return None  # inconclusive runs are not evidence

        overrides = dict(summary.get("overrides") or {})
        if overrides:
            action = {"type": "config", "payload": overrides}
            action_text = ", ".join("%s=%s" % (k, v) for k, v in sorted(overrides.items()))
        else:
            action = {
                "type": "candidate",
                "payload": {
                    "scenario_id": scenario.id,
                    "candidate_config": summary.get("candidate_config"),
                },
            }
            action_text = "candidate as specified"

        stats = summary.get("stats") or {}
        delta = stats.get("improvement_pct")
        metric = summary.get("primary_metric") or {}
        metric_key = metric.get("key") or scenario.primary_metric.key
        reasons = "; ".join(summary.get("reasons") or [])

        if decision == "PASS":
            claim = "%s: %s improves %s by %+.2f%% under this scope" % (
                scenario.id, action_text, metric_key, delta or 0.0)
            expected_delta = float(delta or 0.0)
            confidence = "candidate"
            kind = "strategy"
        else:
            claim = "%s: %s rejected (%s)" % (scenario.id, action_text, reasons or "gate failure")
            expected_delta = 0.0
            confidence = "refuted"
            kind = "negative"

        scope = {
            "hardware": [scenario.hardware.get("device_model") or scenario.hardware.get("soc") or "unknown"],
            "stack": dict(scenario.stack or {}),
            "model": [scenario.model] if scenario.model else [],
            "topology": [scenario.topology] if scenario.topology else [],
            "workload": {
                k: scenario.workload.get(k)
                for k in ("input_len", "output_len", "concurrency")
                if scenario.workload.get(k) is not None
            },
            "dtypes": [scenario.metadata.get("dtype")] if scenario.metadata.get("dtype") else [],
            "tags": list(scenario.metadata.get("tags") or []),
        }

        card = {
            "kind": kind,
            "claim": claim,
            "scope": scope,
            "action": action,
            "expected": {"metric_key": metric_key, "delta_pct": round(expected_delta, 3)},
            "confidence": confidence,
            "source": "run:%s" % summary.get("run_id"),
            "tags": list(scenario.metadata.get("tags") or []),
            "stats": {
                "applied": 1,
                "success": 1 if decision == "PASS" else 0,
                "fail": 1 if decision == "FAIL" else 0,
                "inconclusive": 0,
                "ema_delta": round(float(delta), 4) if decision == "PASS" and delta is not None else None,
                "calibration_mae": None,
            },
            "evidence": [{
                "run_id": summary.get("run_id"),
                "decision": decision,
                "improvement_pct": delta,
                "ci_low_pct": stats.get("ci_low_pct"),
                "ci_high_pct": stats.get("ci_high_pct"),
                "wins": stats.get("wins"),
                "n": stats.get("n"),
                "overrides": overrides,
                "reasons": summary.get("reasons"),
                "at": utc_now(),
            }],
        }

        # update an existing card with the same action x scope instead of duplicating it
        existing = None
        key = self._card_key(card)
        for current in self.cards.values():
            if current.get("key") == key:
                existing = current
                break

        if existing is None:
            return self.add_card(card)

        evidence = existing.setdefault("evidence", [])
        run_id = summary.get("run_id")
        if not any(e.get("run_id") == run_id for e in evidence):
            evidence.append(card["evidence"][0])
        evidence[:] = evidence[-40:]

        stats_existing = existing.setdefault("stats", _empty_stats())
        stats_existing["applied"] = int(stats_existing.get("applied") or 0) + 1
        if decision == "PASS":
            stats_existing["success"] = int(stats_existing.get("success") or 0) + 1
            prev = stats_existing.get("ema_delta")
            stats_existing["ema_delta"] = round(delta, 4) if prev is None else round(0.6 * float(prev) + 0.4 * float(delta), 4)
        else:
            stats_existing["fail"] = int(stats_existing.get("fail") or 0) + 1

        expected = float((existing.get("expected") or {}).get("delta_pct") or 0.0)
        if delta is not None:
            err = abs(expected - float(delta))
            prev = stats_existing.get("calibration_mae")
            stats_existing["calibration_mae"] = round(err, 4) if prev is None else round(0.6 * float(prev) + 0.4 * err, 4)

        if decision == "PASS" and int(stats_existing.get("success") or 0) >= 2:
            existing["confidence"] = "verified"
        elif decision == "FAIL" and int(stats_existing.get("fail") or 0) >= 2:
            existing["confidence"] = "refuted"
        else:
            existing["confidence"] = "candidate"
        existing["updated"] = utc_now()
        self.save()
        return existing

    # ------------------------------------------------------------------ render
    def render_priors(self, matched: list, max_claim: int = 240) -> str:
        if not matched:
            return "## 调优先验\n（知识库中没有条件匹配的记录——这轮需要自己建立证据）"
        lines = ["## 调优先验（按硬件/模型/版本/负载条件检索，按历史证据排序）"]
        for card, score, match in matched:
            claim = str(card.get("claim") or "")
            if len(claim) > max_claim:
                claim = claim[:max_claim] + "…"
            scope = card.get("scope") or {}
            bits = []
            if scope.get("hardware"):
                bits.append("hw=%s" % ",".join(scope["hardware"]))
            if scope.get("model"):
                bits.append("model=%s" % ",".join(scope["model"]))
            if scope.get("topology"):
                bits.append("topo=%s" % ",".join(scope["topology"]))
            stats = card.get("stats") or {}
            history = "%d/%d ok" % (int(stats.get("success") or 0), int(stats.get("applied") or 0))
            expected = card.get("expected") or {}
            exp = ""
            if expected.get("metric_key"):
                exp = " | expected %+.1f%% %s" % (
                    float(expected.get("delta_pct") or 0.0), expected.get("metric_key"))
            marker = "NEGATIVE" if card.get("kind") == "negative" else card.get("kind", "strategy")
            lines.append("- [%s] (%s) %s" % (card.get("id"), marker, claim))
            lines.append("    %s | history %s | confidence %s%s | match %.2f" % (
                " | ".join(bits) or "scope=any", history,
                card.get("confidence"), exp, match))
        return "\n".join(lines)