"""Safety limits for diagnostics: tests never start a model worker."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnostics/profile_mps.py"
spec = importlib.util.spec_from_file_location("profile_mps", SCRIPT)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class DiagnosticLimitsTests(unittest.TestCase):
    def test_round_allowance_is_idempotent_and_separate(self):
        previous = {"charged_seconds": 272.22}
        ledger = {"workers": [previous]}
        for _ in range(2):
            self.assertEqual(profile.round_allowance(ledger, "nar-cost"), 300)
        self.assertEqual(list(ledger["grants"]), [profile.GRANT])
        ledger["workers"].append({"grant": profile.GRANT, "charged_seconds": 260})
        self.assertEqual(profile.round_allowance(ledger, "ar-combo"), 40)
        self.assertAlmostEqual(profile.round_allowance(ledger, "ar"), 27.78)
        self.assertEqual(ledger["workers"][0], previous)

    def test_new_grant_requires_original_existing_ledger(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(profile, "ROOT", Path(directory)):
            canonical = Path(directory) / "runs/probes/mps-performance"
            for path in (canonical, Path(directory) / "other"):
                with self.subTest(path=path), patch.object(profile.subprocess, "run") as launch:
                    with self.assertRaisesRegex(SystemExit, "original shared MPS ledger"):
                        profile.launch(argparse.Namespace(output=path, run=path, mode="nar-cost"))
                    launch.assert_not_called()

    def test_candidate_uses_same_ledger_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = {"ended_at": 0, "charged_seconds": 255}
            (root / "budget.json").write_text(json.dumps({"workers": [previous]}))
            interpreter = root / "candidate/bin/python"
            def execute(command, **kwargs):
                if command[0] == "pmset":
                    return subprocess.CompletedProcess(command, 0, stdout="")
                self.assertEqual(command[0], str(interpreter))
                self.assertGreater(kwargs["timeout"], 44)
                self.assertLessEqual(kwargs["timeout"], 45)
                self.assertEqual(kwargs["env"]["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
                return subprocess.CompletedProcess(command, 0)
            with patch.object(profile.subprocess, "run", side_effect=execute):
                profile.launch(argparse.Namespace(output=root, run=root, mode="separator", python=interpreter, deadline=45))
            ledger = json.loads((root / "budget.json").read_text())
            self.assertEqual(ledger["workers"][0], previous)
            self.assertEqual(len(ledger["workers"]), 2)
            self.assertEqual(ledger["workers"][1]["status"], "passed")

    def test_cooldown_prevents_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "budget.json").write_text(json.dumps({"workers": [
                {"ended_at": time.time(), "charged_seconds": 1}]}))
            with patch.object(profile.subprocess, "run") as launch:
                with self.assertRaisesRegex(SystemExit, "Cooling interval"):
                    profile.launch(argparse.Namespace(output=root, run=root, mode="nar"))
                launch.assert_not_called()

    def test_budget_prevents_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "budget.json").write_text(json.dumps({"workers": [
                {"ended_at": 0, "charged_seconds": 250}]}))
            with patch.object(profile.subprocess, "run") as launch:
                with self.assertRaisesRegex(SystemExit, "budget"):
                    profile.launch(argparse.Namespace(output=root, run=root, mode="nar"))
                launch.assert_not_called()

    def failure_case(self, timeout):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def execute(command, **kwargs):
                if command[0] == "pmset":
                    return subprocess.CompletedProcess(command, 0, stdout="No thermal measurement\n")
                self.assertGreater(kwargs["timeout"], 29)
                self.assertLessEqual(kwargs["timeout"], 30)
                if timeout:
                    raise subprocess.TimeoutExpired(command, 60)
                return subprocess.CompletedProcess(command, 2)
            with patch.object(profile.subprocess, "run", side_effect=execute):
                with self.assertRaises(SystemExit) as caught:
                    profile.launch(argparse.Namespace(output=root, run=root, mode="separator", deadline=30))
                self.assertEqual(caught.exception.code, 1)
            ledger = json.loads((root / "budget.json").read_text())
            self.assertEqual(len(ledger["workers"]), 1)
            self.assertEqual(ledger["workers"][0]["status"], "deadline_exceeded" if timeout else "failed")

    def test_timeout_is_recorded_and_exits_nonzero(self):
        self.failure_case(timeout=True)

    def test_worker_failure_is_recorded_and_exits_nonzero(self):
        self.failure_case(timeout=False)
