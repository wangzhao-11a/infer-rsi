"""Stdlib-only tests for InferTune-RSI.

Run from the package root:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from infer_rsi import gates, stats
from infer_rsi.cli import main
from infer_rsi.harness import run_scenario, sweep
from infer_rsi.knowledge import KnowledgeStore
from infer_rsi.spec import Constraint, MetricSpec, SpecError, load_scenario, scenario_from_dict

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = os.path.join(PKG_ROOT, "scenarios")
KNOWLEDGE = os.path.join(PKG_ROOT, "knowledge")


def scenario_path(name: str) -> str:
    return os.path.join(SCENARIOS, name)


def loaded(name: str):
    return load_scenario(scenario_path(name))


class SpecTests(unittest.TestCase):
    def test_requires_result_placeholder(self):
        data = {
            "id": "x",
            "workdir": os.getcwd(),
            "primary_metric": {"key": "m"},
            "baseline_command": "python a.py",
            "candidate_command": "python b.py {result}",
        }
        with self.assertRaises(SpecError):
            scenario_from_dict(data)

    def test_rejects_bad_direction(self):
        data = {
            "id": "x",
            "workdir": os.getcwd(),
            "primary_metric": {"key": "m", "direction": "sideways"},
            "baseline_command": "python a.py {result}",
            "candidate_command": "python b.py {result}",
        }
        with self.assertRaises(SpecError):
            scenario_from_dict(data)

    def test_resolves_workdir_relative_to_scenario(self):
        scenario = loaded("demo_pass.json")
        self.assertEqual(os.path.abspath(scenario.workdir), PKG_ROOT)


class StatsTests(unittest.TestCase):
    def test_improvement_direction(self):
        self.assertAlmostEqual(stats.improvement_pct(100, 110, "maximize"), 10.0)
        self.assertAlmostEqual(stats.improvement_pct(110, 100, "minimize"), 10.0)
        self.assertAlmostEqual(stats.improvement_pct(100, 90, "maximize"), -10.0)

    def test_bootstrap_is_deterministic(self):
        values = [1.0, 2.0, 3.0, 4.0]
        first = stats.bootstrap_ci(values, seed=7)
        second = stats.bootstrap_ci(values, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])

    def test_cv(self):
        self.assertAlmostEqual(stats.cv_pct([100, 100, 100]), 0.0, places=6)
        self.assertGreater(stats.cv_pct([80, 100, 120]), 10.0)


def fake_scenario(quality_command="q", constraints=None, min_improvement=3.0, noise_cv=8.0):
    return SimpleNamespace(
        primary_metric=MetricSpec("throughput", "tok/s", "maximize"),
        constraints=constraints if constraints is not None else [Constraint("tpot", "<=", 35)],
        quality_command=quality_command,
        min_improvement_pct=min_improvement,
        noise_cv_pct=noise_cv,
    )


class GateTests(unittest.TestCase):
    def _evaluate(self, base, cand, quality=None, scenario=None):
        scenario = scenario or fake_scenario()
        quality = quality if quality is not None else [True] * len(base)
        return gates.evaluate(scenario, base, cand, quality)

    def test_pass(self):
        base = [{"throughput": 1000, "tpot": 30}] * 3
        cand = [{"throughput": 1080, "tpot": 30}] * 3
        result = self._evaluate(base, cand)
        self.assertEqual(result["decision"], "PASS")
        self.assertAlmostEqual(result["stats"]["improvement_pct"], 8.0, places=3)

    def test_quality_failure_blocks(self):
        base = [{"throughput": 1000, "tpot": 30}] * 3
        cand = [{"throughput": 1300, "tpot": 30}] * 3
        result = self._evaluate(base, cand, quality=[True, False, True])
        self.assertEqual(result["decision"], "FAIL")
        self.assertTrue(any("quality" in r for r in result["reasons"]))

    def test_constraint_failure_blocks(self):
        base = [{"throughput": 1000, "tpot": 30}] * 3
        cand = [{"throughput": 1300, "tpot": 45}] * 3
        result = self._evaluate(base, cand)
        self.assertEqual(result["decision"], "FAIL")
        self.assertTrue(any("constraint" in r for r in result["reasons"]))

    def test_noise_makes_inconclusive(self):
        base = [{"throughput": 800, "tpot": 30}, {"throughput": 1200, "tpot": 30},
                {"throughput": 1000, "tpot": 30}]
        cand = [{"throughput": 864, "tpot": 30}, {"throughput": 1296, "tpot": 30},
                {"throughput": 1080, "tpot": 30}]
        result = self._evaluate(base, cand)
        self.assertEqual(result["decision"], "INCONCLUSIVE")
        self.assertTrue(any("noise" in r for r in result["reasons"]))

    def test_worse_is_fail(self):
        base = [{"throughput": 1000, "tpot": 30}] * 3
        cand = [{"throughput": 900, "tpot": 30}] * 3
        result = self._evaluate(base, cand)
        self.assertEqual(result["decision"], "FAIL")
        self.assertTrue(any("worse" in r for r in result["reasons"]))

    def test_below_bar_is_fail(self):
        base = [{"throughput": 1000, "tpot": 30}] * 3
        cand = [{"throughput": 1010, "tpot": 30}] * 3
        result = self._evaluate(base, cand)
        self.assertEqual(result["decision"], "FAIL")
        self.assertTrue(any("below" in r for r in result["reasons"]))

    def test_partial_reps_are_inconclusive(self):
        base = [{"throughput": 1000, "tpot": 30}]
        cand = [{"throughput": 1080, "tpot": 30}]
        result = gates.evaluate(fake_scenario(), base, cand, [True], expected_n=3)
        self.assertEqual(result["decision"], "INCONCLUSIVE")
        self.assertTrue(any("planned repetitions" in r for r in result["reasons"]))


class KnowledgeTests(unittest.TestCase):
    def test_feasibility_blocker(self):
        store = KnowledgeStore(KNOWLEDGE)
        blockers, _ = store.check_feasibility(loaded("demo_blocked.json"))
        ids = {card["id"] for card in blockers}
        self.assertIn("kc-0001", ids)

    def test_model_filtered_from_priors(self):
        store = KnowledgeStore(KNOWLEDGE)
        matched = store.retrieve(loaded("demo_pass.json"), top_k=20)
        ids = {card["id"] for card, _score, _match in matched}
        self.assertNotIn("kc-0004", ids)  # scoped to another model
        self.assertIn("kc-0002", ids)

    def test_distill_upgrades_after_two_verified_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = os.path.join(tmp, "kb")
            runs = os.path.join(tmp, "runs")
            scenario = loaded("demo_knob.json")
            first = run_scenario(scenario, overrides={"max_num_seqs": 108},
                                 ledger_root=runs, knowledge_root=kb, distill=True, quiet=True)
            self.assertEqual(first["decision"], "PASS")
            second = run_scenario(scenario, overrides={"max_num_seqs": 108},
                                  ledger_root=runs, knowledge_root=kb, distill=True, quiet=True)
            self.assertEqual(second["decision"], "PASS")
            store = KnowledgeStore(kb)
            cards = [c for c in store.cards.values() if c.get("kind") == "strategy"]
            self.assertEqual(len(cards), 1)
            card = cards[0]
            self.assertEqual(card["stats"]["applied"], 2)
            self.assertEqual(card["stats"]["success"], 2)
            self.assertEqual(card["confidence"], "verified")
            self.assertEqual(len(card["evidence"]), 2)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="infer-rsi-test-")
        self.runs = os.path.join(self.tmp, "runs")
        self.kb = os.path.join(self.tmp, "kb")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_scenario_file(self, name, **kwargs):
        return run_scenario(loaded(name), ledger_root=self.runs,
                            knowledge_root=self.kb, quiet=True, **kwargs)

    def test_pass_writes_report_and_ledger(self):
        summary = self.run_scenario_file("demo_pass.json")
        self.assertEqual(summary["decision"], "PASS")
        self.assertGreater(summary["stats"]["improvement_pct"], 3.0)
        self.assertTrue(os.path.isfile(summary["report_path"]))
        ledger_path = os.path.join(self.runs, "ledger.jsonl")
        self.assertTrue(os.path.isfile(ledger_path))
        with open(ledger_path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        self.assertEqual(rows[-1]["decision"], "PASS")

    def test_quality_failure(self):
        summary = self.run_scenario_file("demo_quality_fail.json")
        self.assertEqual(summary["decision"], "FAIL")
        self.assertTrue(any("quality" in r for r in summary["reasons"]))

    def test_noisy_is_inconclusive(self):
        summary = self.run_scenario_file("demo_noisy.json")
        self.assertEqual(summary["decision"], "INCONCLUSIVE")

    def test_constraint_failure(self):
        summary = self.run_scenario_file("demo_constraint_fail.json")
        self.assertEqual(summary["decision"], "FAIL")
        self.assertTrue(any("constraint" in r for r in summary["reasons"]))

    def test_worse_is_fail(self):
        summary = self.run_scenario_file("demo_worse.json")
        self.assertEqual(summary["decision"], "FAIL")

    def test_sweep_finds_108(self):
        result = sweep(loaded("demo_knob.json"), "max_num_seqs", [72, 90, 108],
                       ledger_root=self.runs, knowledge_root=self.kb,
                       repetitions=3, quiet=True)
        self.assertIsNotNone(result["best_pass"])
        self.assertEqual(result["best_pass"]["value"], 108)
        by_value = {row["value"]: row for row in result["rows"]}
        self.assertNotEqual(by_value[72]["decision"], "PASS")


class CliTests(unittest.TestCase):
    def test_doctor_blocks_demo(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["doctor", "--scenario", scenario_path("demo_blocked.json"),
                         "--knowledge-root", KNOWLEDGE])
        self.assertEqual(code, 1)
        self.assertIn("kc-0001", out.getvalue())

    def test_plan_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["plan", "--scenario", scenario_path("demo_pass.json"),
                             "--json", "--knowledge-root", KNOWLEDGE,
                             "--ledger-root", os.path.join(tmp, "runs")])
            self.assertEqual(code, 0)
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["plan"])
            self.assertTrue(payload["priors"])

    def test_run_dry_run(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["run", "--scenario", scenario_path("demo_pass.json"), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", out.getvalue())


if __name__ == "__main__":
    unittest.main()