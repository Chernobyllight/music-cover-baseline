import os
import json
import tempfile
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from music_cover.runtime import ForwardAudit, memory_budget, resolve_device


class RuntimeTests(unittest.TestCase):
    def test_no_gpu_never_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            for device in ("auto", "mps", "cuda"):
                with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                    resolve_device(device)

    def test_explicit_device_and_auto_precedence(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.backends.mps.is_available", return_value=True), patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}):
            self.assertEqual(resolve_device("auto"), "cuda")
            self.assertEqual(resolve_device("mps"), "mps")

    def test_reject_implicit_mps_fallback(self):
        with patch("torch.backends.mps.is_available", return_value=True), patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(RuntimeError, "before importing torch"):
                resolve_device("mps")

    def test_memory_configuration(self):
        self.assertEqual(memory_budget(None, "mps"), 16)
        self.assertEqual(memory_budget(None, "cuda"), 80)
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                memory_budget(invalid, "mps")

    def test_audit_rejects_actual_cpu_model_and_removes_hook(self):
        import torch
        layer = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(RuntimeError, "forward on cpu"):
            with ForwardAudit("mps"):
                layer(torch.ones(1, 2))
        self.assertEqual(tuple(layer(torch.ones(1, 2)).shape), (1, 2))

    def test_cli_rejects_gpu_mps_conflict_before_loading_models(self):
        result = subprocess.run([sys.executable, "-m", "music_cover.pipeline", "--preflight",
                                 "--device", "mps", "--gpu", "0"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--gpu cannot be combined", result.stderr)

    def test_resume_cuda_run_can_explicitly_select_mac_device(self):
        from music_cover import pipeline
        for device in ("mps", "auto"):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "request.json").write_text(json.dumps({"gpu": "0", "memory_budget_gib": 80}))
                def inspect(request):
                    self.assertEqual(request["device"], device)
                    self.assertIsNone(request["gpu"])
                    self.assertIsNone(request["memory_budget_gib"])
                    return {}
                with patch.object(sys, "argv", ["pipeline", "--resume", str(root), "--device", device, "--preflight"]), patch.object(pipeline, "preflight", side_effect=inspect):
                    pipeline.main()

    def test_upstream_identity_change_invalidates_cached_stage(self):
        from music_cover.pipeline import Pipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("input")
            request = {"device": "mps", "runtime": {"upstream": {"source_sha256": "before"}}}
            pipeline = Pipeline(root / "run", request)
            calls = []
            def action(out):
                calls.append(out)
                path = out / "result.txt"
                path.write_text("result")
                return {"output": path}
            try:
                pipeline.stage("model", {"input": source}, {}, action)
                pipeline.stage("model", {"input": source}, {}, action)
                self.assertEqual(len(calls), 1)
                request["runtime"]["upstream"]["source_sha256"] = "after"
                pipeline.stage("model", {"input": source}, {}, action)
                self.assertEqual(len(calls), 2)
            finally:
                pipeline.lock.close()


if __name__ == "__main__":
    unittest.main()
