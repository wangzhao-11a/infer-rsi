"""Command line entry point for InferTune-RSI.

    python -m infer_rsi doctor  --scenario scenarios/demo_pass.json
    python -m infer_rsi plan    --scenario scenarios/demo_pass.json
    python -m infer_rsi run     --scenario scenarios/demo_pass.json --distill
    python -m infer_rsi sweep   --scenario scenarios/demo_knob.json --knob max_num_seqs --values 72,90,108
    python -m infer_rsi report  --latest
    python -m infer_rsi knowledge search --scenario scenarios/demo_pass.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .catalog import get_knob
from .harness import (
    BlockedError, LockError, command_available, run_scenario, sweep,
)
from .knowledge import DEFAULT_KNOWLEDGE_ROOT, KnowledgeStore
from .ledger import DEFAULT_LEDGER_ROOT, RunLedger
from .planner import propose, prompt_block, render_plan
from .report import render_blocked
from .spec import SpecError, load_scenario

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCENARIOS = os.path.join(PKG_ROOT, "scenarios")

EXIT_PASS, EXIT_FAIL, EXIT_INCONCLUSIVE, EXIT_ERROR = 0, 1, 2, 3


def _parse_scalar(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def _parse_overrides(pairs) -> dict:
    overrides = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SpecError("--override expects key=value, got %r" % pair)
        key, _, value = pair.partition("=")
        overrides[key.strip()] = _parse_scalar(value.strip())
    return overrides


def _print_decision(summary: dict, quiet: bool = False) -> int:
    decision = summary.get("decision")
    stats = summary.get("stats") or {}
    metric = summary.get("primary_metric") or {}
    if not quiet:
        print("")
        print("结论: %s" % decision)
        if stats.get("improvement_pct") is not None:
            print("主指标 %s: baseline=%s candidate=%s  Δ=%+.2f%%  90%%CI=[%+.2f%%, %+.2f%%]  wins=%s/%s" % (
                metric.get("key"),
                _fmt(stats.get("baseline_median")), _fmt(stats.get("candidate_median")),
                stats.get("improvement_pct"),
                stats.get("ci_low_pct") or 0.0, stats.get("ci_high_pct") or 0.0,
                stats.get("wins"), stats.get("n"),
            ))
        for reason in summary.get("reasons") or []:
            print("  - %s" % reason)
        if summary.get("report_path"):
            print("报告: %s" % summary["report_path"])
        if summary.get("knowledge_card"):
            print("知识卡: %s" % summary["knowledge_card"])
    if decision == "PASS":
        return EXIT_PASS
    if decision == "FAIL":
        return EXIT_FAIL
    return EXIT_INCONCLUSIVE


def _fmt(value) -> str:
    if value is None:
        return "-"
    try:
        return "%.3f" % float(value)
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- commands
def cmd_doctor(args) -> int:
    scenario = load_scenario(args.scenario)
    store = KnowledgeStore(args.knowledge_root)
    print("场景: %s" % scenario.id)
    print("说明: %s" % (scenario.description or "-"))
    print("模型: %s | 拓扑: %s | backend: %s" % (scenario.model, scenario.topology, scenario.backend))
    print("硬件: %s | 栈: %s" % (scenario.hardware, scenario.stack))
    print("负载: %s" % scenario.workload)
    print("配置: %s" % scenario.current_config)
    print("主指标: %s (%s) | 约束: %s" % (
        scenario.primary_metric.key, scenario.primary_metric.direction,
        ["%s%s%s" % (c.key, c.op, c.value) for c in scenario.constraints]))
    print("判定: reps=%d, min_improvement=%.2f%%, noise_cv<=%.2f%%, budget=%.0fmin" % (
        scenario.repetitions, scenario.min_improvement_pct, scenario.noise_cv_pct, scenario.max_minutes))
    print("工作目录: %s" % scenario.workdir)

    errors, warnings = [], []
    if scenario.is_template:
        warnings.append("scenario 标记为 template：命令是模板，不能直接运行")

    for label, command in (("baseline", scenario.baseline_command),
                           ("candidate", scenario.candidate_command),
                           ("quality", scenario.quality_command)):
        if not command:
            continue
        ok, resolved = command_available(command, scenario.workdir)
        print("%-9s 命令: %s" % (label, resolved or command))
        if not ok:
            (warnings if scenario.is_template or label == "quality" else errors).append(
                "%s 命令的可执行文件找不到（%s）" % (label, resolved))

    ledger = RunLedger(args.ledger_root)
    from .ledger import git_fingerprint, env_fingerprint
    git = git_fingerprint(scenario.repo)
    if git.get("available"):
        print("git: %s @ %s%s" % (git.get("branch"), git.get("head", "")[:12],
                                  " (dirty)" if git.get("dirty") else ""))
        if git.get("dirty"):
            warnings.append("工作区有未提交改动；实验记录已保存 diff hash，但候选来源需人工确认")
    else:
        warnings.append("repo 不是 git 工作区：无法用 SHA 追溯候选来源")
        print("git: 不可用（%s）" % git.get("reason"))
    print("env: %s" % env_fingerprint().get("platform"))

    blockers, allowed = store.check_feasibility(scenario)
    if blockers:
        print("")
        print(render_blocked(blockers, scenario))
    for card in allowed:
        warnings.append("可行性卡 %s: %s" % (card.get("id"), card.get("claim")))

    recent = ledger.latest(scenario.id)
    if recent:
        print("最近一次: %s -> %s (%s)" % (recent.get("run_id"), recent.get("decision"),
                                           _fmt(recent.get("improvement_pct"))))

    if warnings:
        print("")
        print("警告:")
        for item in warnings:
            print("  - %s" % item)
    if errors:
        print("")
        print("错误:")
        for item in errors:
            print("  - %s" % item)
        return EXIT_ERROR
    if blockers:
        return EXIT_FAIL
    print("")
    print("OK: 场景可执行")
    return EXIT_PASS


def cmd_plan(args) -> int:
    scenario = load_scenario(args.scenario)
    store = KnowledgeStore(args.knowledge_root)
    blockers, _ = store.check_feasibility(scenario)
    if blockers and not args.allow_blocked:
        print(render_blocked(blockers, scenario))
        return EXIT_FAIL

    matched = store.retrieve(scenario, top_k=args.top)
    plan = propose(scenario, store, top_k=args.top)
    priors_text = store.render_priors(matched)

    if args.prompt:
        print(prompt_block(scenario, plan, priors_text))
        return EXIT_PASS
    if args.json:
        print(json.dumps({
            "scenario": scenario.to_dict(),
            "priors": [
                {"id": card.get("id"), "claim": card.get("claim"), "score": score,
                 "match": match, "confidence": card.get("confidence")}
                for card, score, match in matched
            ],
            "plan": plan,
        }, ensure_ascii=False, indent=2))
        return EXIT_PASS

    print("# 场景 `%s`" % scenario.id)
    print("模型=%s 拓扑=%s 硬件=%s" % (
        scenario.model, scenario.topology,
        scenario.hardware.get("device_model") or scenario.hardware.get("soc")))
    print("")
    print(priors_text)
    print("")
    print(render_plan(scenario, plan))
    return EXIT_PASS


def cmd_run(args) -> int:
    scenario = load_scenario(args.scenario)
    overrides = _parse_overrides(args.override)
    if args.override_json:
        try:
            overrides.update(json.loads(args.override_json))
        except ValueError as exc:
            raise SpecError("--override-json is not valid JSON: %s" % exc)

    try:
        summary = run_scenario(
            scenario, overrides=overrides, ledger_root=args.ledger_root,
            knowledge_root=args.knowledge_root, dry_run=args.dry_run,
            distill=args.distill, allow_blocked=args.allow_blocked,
            tag=args.tag, quiet=args.quiet, repetitions=args.reps,
        )
    except BlockedError as exc:
        print(render_blocked(exc.blockers, scenario))
        return EXIT_FAIL
    except LockError as exc:
        print("锁冲突: %s" % exc, file=sys.stderr)
        return EXIT_ERROR

    if args.dry_run:
        print("DRY RUN: %s" % scenario.id)
        print("baseline : %s" % summary["commands"]["baseline"])
        print("candidate: %s" % summary["commands"]["candidate"])
        if summary["commands"].get("quality"):
            print("quality  : %s" % summary["commands"]["quality"])
        print("candidate config: %s" % json.dumps(summary["candidate_config"], ensure_ascii=False))
        return EXIT_PASS
    return _print_decision(summary, quiet=args.quiet)


def cmd_sweep(args) -> int:
    scenario = load_scenario(args.scenario)
    knob = get_knob(args.knob)
    if not knob:
        raise SpecError("unknown knob %r; see infer_rsi/catalog.py" % args.knob)
    values = [_parse_scalar(v.strip()) for v in args.values.split(",") if v.strip()]
    if not values:
        raise SpecError("--values must contain at least one value")
    result = sweep(
        scenario, args.knob, values, ledger_root=args.ledger_root,
        knowledge_root=args.knowledge_root, repetitions=args.reps,
        tag=args.tag, quiet=args.quiet,
    )
    print("")
    print("| %s | baseline | candidate | Δ%% | CI | wins | decision |" % args.knob)
    print("|---|---:|---:|---:|---|---:|---|")
    for row in result["rows"]:
        print("| %s | %s | %s | %+.2f | %s..%s | %s/%s | %s |" % (
            row["value"], _fmt(row["baseline"]), _fmt(row["primary"]),
            row["improvement_pct"] or 0.0, _fmt(row["ci_low_pct"]), _fmt(row["ci_high_pct"]),
            row["wins"], row["n"], row["decision"]))
    if result.get("best_pass"):
        print("最佳且通过门禁: %s (%+.2f%%)" % (
            result["best_pass"]["value"], result["best_pass"]["improvement_pct"]))
    print("sweep 目录: %s" % result["sweep_dir"])
    if result.get("best_pass"):
        return EXIT_PASS
    return EXIT_INCONCLUSIVE


def cmd_report(args) -> int:
    ledger = RunLedger(args.ledger_root)
    if args.run_id:
        run_dir = ledger.find(args.run_id)
    else:
        row = ledger.latest(args.scenario)
        run_dir = ledger.find(row["run_id"]) if row else None
    if not run_dir:
        print("找不到运行记录", file=sys.stderr)
        return EXIT_ERROR
    report = ledger.read_report(run_dir)
    if not report:
        print("运行目录里没有 report.md: %s" % run_dir, file=sys.stderr)
        return EXIT_ERROR
    print(report)
    return EXIT_PASS


def cmd_knowledge(args) -> int:
    store = KnowledgeStore(args.knowledge_root)
    if args.action == "add":
        with open(args.file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        cards = payload if isinstance(payload, list) else [payload]
        for card in cards:
            record = store.add_card(card)
            print("upserted %s" % record.get("id"))
        return EXIT_PASS

    if args.action == "list":
        cards = list(store.cards.values())
        if args.kind:
            cards = [c for c in cards if c.get("kind") == args.kind]
        cards.sort(key=lambda c: c.get("id", ""))
        for card in cards:
            print("%-9s %-11s %-10s %s" % (
                card.get("id"), card.get("kind"), card.get("confidence"),
                str(card.get("claim") or "")[:110]))
        print("\n共 %d 张" % len(cards))
        return EXIT_PASS

    if args.action == "search":
        if args.scenario:
            scenario = load_scenario(args.scenario)
            matched = store.retrieve(scenario, query=args.query or "", top_k=args.top)
            if args.json:
                print(json.dumps([
                    {"id": c.get("id"), "claim": c.get("claim"), "score": s, "match": m,
                     "confidence": c.get("confidence"), "kind": c.get("kind")}
                    for c, s, m in matched
                ], ensure_ascii=False, indent=2))
            else:
                print(store.render_priors(matched))
        else:
            cards = store.search(args.query or "", top_k=args.top)
            for card in cards:
                print("- [%s] (%s) %s" % (card.get("id"), card.get("kind"), card.get("claim")))
        return EXIT_PASS
    return EXIT_ERROR


def cmd_demo(args) -> int:
    path = os.path.join(DEFAULT_SCENARIOS, "demo_pass.json")
    print("demo scenario: %s" % path)
    scenario = load_scenario(path)
    summary = run_scenario(
        scenario, ledger_root=args.ledger_root, knowledge_root=args.knowledge_root,
        distill=args.distill, quiet=args.quiet,
    )
    return _print_decision(summary, quiet=args.quiet)


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer_rsi",
        description="InferTune-RSI: A/B experiment ledger + gates + knowledge distillation for inference tuning")
    parser.add_argument("--version", action="version", version="infer-rsi %s" % __version__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ledger-root", default=DEFAULT_LEDGER_ROOT)
    common.add_argument("--knowledge-root", default=DEFAULT_KNOWLEDGE_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", parents=[common], help="validate a scenario before spending GPU time")
    doctor.add_argument("--scenario", required=True)
    doctor.set_defaults(func=cmd_doctor)

    plan = sub.add_parser("plan", parents=[common], help="retrieve priors and rank allow-listed hypotheses")
    plan.add_argument("--scenario", required=True)
    plan.add_argument("--top", type=int, default=8)
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--prompt", action="store_true", help="emit a self-contained prompt for an agent")
    plan.add_argument("--allow-blocked", action="store_true")
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", parents=[common], help="run baseline vs candidate and decide")
    run.add_argument("--scenario", required=True)
    run.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--override-json", default="")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--distill", action="store_true", help="write a knowledge card from the result")
    run.add_argument("--allow-blocked", action="store_true")
    run.add_argument("--reps", type=int, default=None)
    run.add_argument("--tag", default="")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)

    sw = sub.add_parser("sweep", parents=[common], help="run one knob over several values")
    sw.add_argument("--scenario", required=True)
    sw.add_argument("--knob", required=True)
    sw.add_argument("--values", required=True, help="comma-separated, e.g. 72,90,108")
    sw.add_argument("--reps", type=int, default=None)
    sw.add_argument("--tag", default="sweep")
    sw.add_argument("--quiet", action="store_true")
    sw.set_defaults(func=cmd_sweep)

    report = sub.add_parser("report", parents=[common], help="print a run report")
    report.add_argument("--run-id", default="")
    report.add_argument("--latest", action="store_true")
    report.add_argument("--scenario", default="")
    report.set_defaults(func=cmd_report)

    knowledge = sub.add_parser("knowledge", parents=[common], help="inspect or add knowledge cards")
    knowledge.add_argument("action", choices=["search", "add", "list"])
    knowledge.add_argument("--scenario", default="")
    knowledge.add_argument("--query", default="")
    knowledge.add_argument("--top", type=int, default=10)
    knowledge.add_argument("--kind", default="")
    knowledge.add_argument("--file", default="")
    knowledge.add_argument("--json", action="store_true")
    knowledge.set_defaults(func=cmd_knowledge)

    demo = sub.add_parser("demo", parents=[common], help="run the synthetic end-to-end demo")
    demo.add_argument("--distill", action="store_true")
    demo.add_argument("--quiet", action="store_true")
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SpecError as exc:
        print("场景错误: %s" % exc, file=sys.stderr)
        return EXIT_ERROR
    except (LockError, BlockedError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        print("文件不存在: %s" % exc, file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())