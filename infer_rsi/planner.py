"""Planner: turn a scenario + retrieved priors into ranked, allow-listed hypotheses.

The planner is deliberately not an LLM.  It is a deterministic, reviewable
mutation generator over the knob catalog.  An LLM can be
inserted in front of it: it reads `prompt_block()`, proposes one of these
mutations or a patch, and the harness still validates the result.  That split is
what keeps "self-evolution" from turning into unreviewed production edits.
"""
from __future__ import annotations

from .catalog import KNOBS, knob_applies

COST_PENALTY = {"restart": 0.05, "topology": 0.25, "patch": 0.35}
RISK_PENALTY = {"low": 0.0, "medium": 0.05, "high": 0.15}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _apply_mutation(current, mutation: dict):
    kind = mutation.get("kind")
    if kind == "none":
        return None
    if kind == "toggle":
        return not bool(current) if current is not None else True
    if kind == "scale":
        if not _is_number(current):
            return None
        value = current * float(mutation["factor"])
        if isinstance(current, int):
            value = int(round(value))
            if value == current:
                value = current + (1 if mutation["factor"] > 1 else -1)
        return value
    if kind == "enum":
        values = list(mutation.get("values") or [])
        return [v for v in values if v != current] or None
    return None


def _expanded_values(knob: dict, current):
    values = []
    for mutation in knob.get("mutations") or []:
        if mutation.get("kind") == "enum":
            values.extend(_apply_mutation(current, mutation) or [])
        else:
            value = _apply_mutation(current, mutation)
            if value is not None:
                values.append(value)
    seen, out = set(), []
    for value in values:
        marker = repr(value)
        if marker not in seen:
            seen.add(marker)
            out.append(value)
    return out


def _card_relevant(card: dict, knob_key: str) -> bool:
    """A prior only informs the expected effect if it talks about this knob."""
    action = card.get("action") or {}
    payload = action.get("payload") or {}
    if isinstance(payload, dict) and knob_key in payload:
        return True
    haystack = " ".join([
        str(card.get("claim") or ""),
        " ".join(str(t) for t in (card.get("tags") or [])),
        str(card.get("source") or ""),
    ]).lower()
    return knob_key.lower() in haystack or knob_key.replace("_", " ") in haystack


def propose(scenario, knowledge, top_k: int = 8, attempted=()) -> list:
    attempted = set(attempted)
    candidates = []
    for knob in KNOBS:
        if not knob_applies(knob, scenario):
            continue
        current = (scenario.current_config or {}).get(knob["key"])
        for new_value in _expanded_values(knob, current):
            if new_value == current:
                continue
            overrides = {knob["key"]: new_value}
            card_rows = knowledge.retrieve(scenario, query=knob["key"], top_k=4)
            relevant = [row for row in card_rows if _card_relevant(row[0], knob["key"])]
            if relevant:
                prior_score = relevant[0][1]
                prior_cards = [c.get("id") for c, _s, _m in relevant[:3]]
            else:
                prior_score = 0.0
                prior_cards = []
            expected = knob.get("expected_pct", 0.0)
            for card, _score, _match in relevant:
                expected_card = (card.get("expected") or {}).get("delta_pct")
                if expected_card is not None:
                    expected = float(expected_card)
                    break
            novelty = 0.0 if (knob["key"], new_value) in attempted else 0.10
            score = (
                0.40 * min(1.0, max(0.0, float(expected) / 20.0))
                + 0.40 * prior_score
                + novelty
                - COST_PENALTY.get(knob.get("cost"), 0.1)
                - RISK_PENALTY.get(knob.get("risk"), 0.05)
            )
            candidates.append({
                "knob": knob["key"],
                "label": knob.get("label", knob["key"]),
                "from": current,
                "to": new_value,
                "overrides": overrides,
                "expected_pct": round(float(expected), 2),
                "cost": knob.get("cost"),
                "risk": knob.get("risk"),
                "guards": list(knob.get("guards") or []),
                "rationale": knob.get("rationale", ""),
                "source": knob.get("source", ""),
                "prior_cards": prior_cards,
                "prior_score": round(prior_score, 3),
                "score": round(score, 4),
            })

    candidates.sort(key=lambda item: -item["score"])
    deduped, seen = [], set()
    for item in candidates:
        key = (item["knob"], repr(item["to"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= top_k:
            break
    return deduped


def render_plan(scenario, plan: list) -> str:
    lines = [
        "## 候选假设（knob 目录内，允许的改动）",
        "| # | knob | from -> to | 预期 | 成本 | 风险 | 先验 |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, item in enumerate(plan, 1):
        lines.append("| %d | `%s` | %s -> %s | %+.1f%% | %s | %s | %s |" % (
            i, item["knob"], item["from"], item["to"], item["expected_pct"],
            item["cost"], item["risk"], ",".join(item["prior_cards"]) or "-"))
    lines.append("")
    lines.append("选择规则：先低风险/低成本的 restart 级改动；每轮只改一个主变量；"
                 "命中后跑复现实验再升级为 verified。")
    return "\n".join(lines)


def prompt_block(scenario, plan: list, priors_text: str) -> str:
    """A self-contained prompt for an external agent."""
    lines = [
        "# 推理调优任务",
        "",
        "## 场景（不可更改，除非先修改 scenario 文件并重新 doctor）",
        "- id: `%s`" % scenario.id,
        "- model: `%s` | topology: `%s` | backend: `%s`" % (scenario.model, scenario.topology, scenario.backend),
        "- hardware: `%s` | stack: `%s`" % (
            scenario.hardware.get("device_model") or scenario.hardware.get("soc") or "-",
            scenario.stack),
        "- workload: `%s`" % scenario.workload,
        "- primary metric: `%s` (%s) | constraints: %s" % (
            scenario.primary_metric.key, scenario.primary_metric.direction,
            ["%s %s %s" % (c.key, c.op, c.value) for c in scenario.constraints]),
        "- bar: improvement >= %.2f%% | noise CV <= %.2f%% | reps=%d" % (
            scenario.min_improvement_pct, scenario.noise_cv_pct, scenario.repetitions),
        "",
        priors_text,
        "",
        "## 允许的动作",
        "只能从下列 knob 变更中选择一个作为本轮假设，或提出一个可回滚的 patch（patch 需单独写明文件、diff 和回滚方式）：",
    ]
    for i, item in enumerate(plan, 1):
        lines.append("%d. `%s`: %s -> %s（预期 %+.1f%%，成本 %s，风险 %s）— %s" % (
            i, item["knob"], item["from"], item["to"], item["expected_pct"],
            item["cost"], item["risk"], item["rationale"]))
    lines.extend([
        "",
        "## 输出协议（只输出一个 JSON 对象）",
        "```json",
        '{"hypothesis": "一句话说明为什么这个改动在当前场景会有效",',
        ' "knob": "<上面的 knob key>", "from": <当前值>, "to": <新值>,',
        ' "expected_delta_pct": <你预测的主指标变化>,',
        ' "quality_risk": "low|medium|high",',
        ' "falsifier": "如果出现什么现象就判定这个假设错误",',
        ' "rollback": "如何回滚"}',
        "```",
        "",
        "硬性要求：不要修改 baseline；不要同时改多个变量；不要在没有质量门禁的情况下换量化/图模式；"
        "不要尝试 feasibility 卡里标记为 blocked 的特性。",
    ])
    return "\n".join(lines)