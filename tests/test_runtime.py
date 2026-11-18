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
                (root / "stage_state.json").write_text(json.dumps({"schema_version": 2}))
                (root / "config.resolved.json").write_text("{}")
                argv = ["pipeline", "--resume", str(root), "--device", device, "--until", "06", "--preflight"]
                with patch.object(sys, "argv", argv), patch.object(pipeline, "resolve_device", return_value="mps"), \
                     patch.object(pipeline, "check_resources") as check:
                    pipeline.main()
                    self.assertEqual(check.call_args.kwargs["device"], "mps")

    def test_task_switch_discards_only_inherited_device_options(self):
        from music_cover.pipeline import apply_request_overrides
        request = {'device': 'cuda', 'gpu': '2', 'memory_budget_gib': 80}
        apply_request_overrides(request, {'device': 'mps', 'memory_budget_gib': 12})
        self.assertEqual(request, {'device': 'mps', 'gpu': None, 'memory_budget_gib': 12})

    def test_explicit_cuda_keeps_compatible_task_options(self):
        from music_cover.pipeline import DEFAULT_REQUEST, apply_request_overrides
        for cli_options, expected_gpu in (({}, '2'), ({'gpu': '1'}, '1')):
            with self.subTest(cli_options=cli_options):
                request = DEFAULT_REQUEST.copy()
                apply_request_overrides(request, {'gpu': '2', 'memory_budget_gib': 24})
                apply_request_overrides(request, {'device': 'cuda', **cli_options})
                self.assertEqual(request['device'], 'cuda')
                self.assertEqual(request['gpu'], expected_gpu)
                self.assertEqual(request['memory_budget_gib'], 24)

    def test_optimized_checkout_preserves_project_relative_inputs(self):
        from music_cover.common.paths import RunPaths
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            data = root / 'data'
            data.mkdir()
            workspace = root / 'optimized'
            workspace.mkdir()
            (workspace / 'data').symlink_to(data)
            paths = RunPaths(workspace, None)
            self.assertEqual(paths.project_relative(data / 'song.wav'), 'data/song.wav')
            self.assertEqual(paths.resolve('data/song.wav'), data / 'song.wav')
            self.assertEqual(paths.project_relative(root / 'other.wav'), str(root / 'other.wav'))

    def test_environment_and_source_change_affect_mps_signature(self):
        from music_cover.common.resources import signature, execution_python
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / 'configs/macos'
            locks.mkdir(parents=True)
            (locks / 'seed-vc.txt').write_text('locked')
            models = root / 'third_party/seed-vc/modules'
            models.mkdir(parents=True)
            impl = models / 'rmvpe.py'
            impl.write_text('original')
            self.assertEqual(execution_python(root, 'seed-vc', 'pitch', 'mps'), root / 'envs/seed-vc/bin/python')
            with patch('music_cover.common.resources.resource_data', return_value={}), \
                 patch('music_cover.common.resources.subprocess.check_output', return_value='{"device":"mps"}'):
                original = signature(root, ['rmvpe'], 'mps')
                impl.write_text('patched')
                self.assertNotEqual(original, signature(root, ['rmvpe'], 'mps'))


if __name__ == "__main__":
    unittest.main()
