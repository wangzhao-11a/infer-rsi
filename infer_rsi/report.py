"""Human-readable experiment report (Markdown).

The report is written next to the raw results so a reviewer can reconstruct the
decision without reading the harness code: what ran, what was measured, which
gate failed, and what to try next.
"""
from __future__ import annotations


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return ("%%.%df" % digits) % float(value)
    except (TypeError, ValueError):
        return str(value)


def _pct(value) -> str:
    if value is None:
        return "-"
    try:
        return "%+.2f%%" % float(value)
    except (TypeError, ValueError):
        return str(value)


def _decision_label(decision: str) -> str:
    return {
        "PASS": "PASS — 保留候选",
        "FAIL": "FAIL — 拒绝候选",
        "INCONCLUSIVE": "INCONCLUSIVE — 证据不足",
        "DRY_RUN": "DRY RUN — 未执行",
    }.get(decision, decision)


def render_report(result: dict, scenario) -> str:
    stats = result.get("stats") or {}
    metric = result.get("primary_metric") or {}
    metric_key = metric.get("key") or scenario.primary_metric.key
    unit = metric.get("unit") or scenario.primary_metric.unit or ""
    lines = [
        "# InferTune-RSI 实验报告",
        "",
        "- **场景**: `%s` — %s" % (scenario.id, scenario.description or ""),
        "- **结论**: **%s**" % _decision_label(result.get("decision", "")),
        "- **运行**: `%s`" % (result.get("run_id") or "-"),
        "- **模型/拓扑**: `%s` / `%s`" % (scenario.model or "-", scenario.topology or "-"),
        "- **硬件/栈**: `%s` / `%s`" % (
            scenario.hardware.get("device_model") or scenario.hardware.get("soc") or "-",
            scenario.stack),
        "- **预算**: %.1f 分钟，重复 %s 次，实际用时 %.1f 秒" % (
            scenario.max_minutes, result.get("repetitions", scenario.repetitions),
            result.get("duration_s") or 0.0),
        "",
    ]
    if result.get("reasons"):
        lines.append("**判定理由**")
        for reason in result["reasons"]:
            lines.append("- %s" % reason)
        lines.append("")

    # ------------------------------------------------------------- primary metric
    lines += [
        "## 1. 主指标 `%s` %s" % (metric_key, ("(%s)" % unit) if unit else ""),
        "",
        "| rep | baseline | candidate | Δ |",
        "|---:|---:|---:|---:|",
    ]
    for row in stats.get("per_rep") or []:
        lines.append("| %s | %s | %s | %s |" % (
            row.get("index"), _fmt(row.get("baseline")), _fmt(row.get("candidate")),
            _pct(row.get("improvement_pct"))))
    if not stats.get("per_rep"):
        lines.append("| - | - | - | - |")
    lines += [
        "",
        "- 中位数：baseline **%s** → candidate **%s**" % (
            _fmt(stats.get("baseline_median")), _fmt(stats.get("candidate_median"))),
        "- 中位提升：**%s**（90%% CI %s .. %s）" % (
            _pct(stats.get("improvement_pct")), _pct(stats.get("ci_low_pct")),
            _pct(stats.get("ci_high_pct"))),
        "- 胜负：%s/%s 个配对 rep（要求至少 %s）" % (
            stats.get("wins", 0), stats.get("n", 0), result.get("need_wins", "-")),
        "- baseline 噪声 CV：%s%%（门限 %.2f%%）" % (
            _fmt(stats.get("baseline_cv_pct"), 2), scenario.noise_cv_pct),
        "- 预先登记的最小提升：%.2f%%" % scenario.min_improvement_pct,
        "",
    ]

    # ------------------------------------------------------------- constraints
    lines += ["## 2. 硬约束（候选侧，逐 rep 取最差值）", ""]
    if result.get("constraints"):
        lines += ["| 指标 | 要求 | 实测最差 | 结果 |", "|---|---:|---:|---|"]
        for row in result["constraints"]:
            lines.append("| `%s` | %s %s | %s | %s |" % (
                row["key"], row["op"], row["limit"], _fmt(row.get("observed")),
                "PASS" if row.get("passed") else "FAIL"))
    else:
        lines.append("（未配置硬约束）")
    lines.append("")

    # ------------------------------------------------------------- quality
    lines += ["## 3. 质量门禁", ""]
    if scenario.quality_command:
        for row in result.get("quality") or []:
            lines.append("- rep %s: %s — %s" % (
                row.get("rep"), "PASS" if row.get("passed") else "FAIL", row.get("detail")))
    else:
        lines.append("（未配置质量命令；本次结论只覆盖性能，不可用于保留精度敏感改动）")
    lines.append("")

    # ------------------------------------------------------------- gates
    lines += ["## 4. 门禁汇总", "", "| 门禁 | 结果 | 说明 |", "|---|---|---|"]
    for gate in result.get("gates") or []:
        lines.append("| %s | %s | %s |" % (
            gate.get("name"), str(gate.get("status")).upper(), gate.get("detail")))
    lines.append("")

    # ------------------------------------------------------------- config
    lines += ["## 5. 配置差异", "", "baseline:"]
    lines.append("```json")
    lines.append(_json(result.get("baseline_config") or {}))
    lines.append("```")
    lines.append("candidate:")
    lines.append("```json")
    lines.append(_json(result.get("candidate_config") or {}))
    lines.append("```")
    if result.get("overrides"):
        lines.append("本轮只变了：`%s`" % _json(result["overrides"]))
    else:
        lines.append("本轮没有配置覆盖：候选差异来自 candidate 命令本身（补丁 / 不同 worktree / 不同启动脚本）。")
    lines.append("")

    # ------------------------------------------------------------- next actions
    lines += ["## 6. 下一步", ""]
    lines += _next_actions(result, scenario)
    lines.append("")

    lines += [
        "## 7. 产物",
        "",
        "- 运行目录：`%s`" % (result.get("run_dir") or "-"),
        "- 原始结果：`reps/rep-<i>-<arm>.json`",
        "- 命令日志：`logs/rep-<i>-<arm>.log`",
        "- 机器可读摘要：`summary.json`",
        "",
    ]
    return "\n".join(lines)


def _json(value) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _next_actions(result: dict, scenario) -> list:
    decision = result.get("decision")
    reasons = " ".join(result.get("reasons") or [])
    if decision == "PASS":
        return [
            "1. 在干净进程上重跑同一候选（replay），两次都 PASS 后把知识卡升级为 `verified`。",
            "2. 只保留一个主变量；如果本轮同时改了命令/补丁，补一个只开该变量的消融。",
            "3. 在不同并发档位复测，确认收益不是单一负载下的偶然。",
        ]
    if decision == "FAIL":
        if "quality" in reasons:
            return [
                "1. 质量门禁失败：先排查输出差异（算子配对/缓存更新/量化误差），不要为了速度保留。",
                "2. 用逐项消融定位是哪一个改动改变了数值，再决定是否缩小改动范围。",
            ]
        if "constraint" in reasons:
            return [
                "1. 先看被违反的约束（TPOT/TTFT/显存），把对应 knob 往回收一档再测。",
                "2. 不要同时放宽约束和改候选；约束必须在实验前固定。",
            ]
        return [
            "1. 候选没有达到预登记门槛，保留 baseline。",
            "2. 把这次失败写成 negative 卡（`infer_rsi knowledge add`），下一轮先排除。",
        ]
    if decision == "INCONCLUSIVE":
        return [
            "1. 提高重复次数（3 → 5），并确认实验期间没有其他任务共享同一组卡。",
            "2. 用 ABBA 顺序复测；若 baseline CV 仍高于门限，换安静时段或换节点。",
            "3. 若效应量本来就小于噪声，放弃这个方向，把预算留给更大的结构改动。",
        ]
    return ["先跑 `doctor` 检查场景与可行性。"]


def render_blocked(blockers: list, scenario) -> str:
    lines = [
        "# 场景被可行性门禁阻断",
        "",
        "场景 `%s` 请求的特性在 `%s` 上已知不可用，未消耗 GPU 时间：" % (
            scenario.id, scenario.hardware.get("device_model") or scenario.hardware.get("soc")),
        "",
    ]
    for card in blockers:
        lines.append("- **%s**: %s" % (card.get("id"), card.get("claim")))
        if card.get("reason"):
            lines.append("  - 原因：%s" % card["reason"])
        if card.get("source"):
            lines.append("  - 证据：%s" % card["source"])
    lines.append("")
    lines.append("如果已在上游版本修复，请先更新知识卡（附新证据），或用 `--allow-blocked` 显式承担风险。")
    return "\n".join(lines)