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
                     patch("music_cover.voice.resolve_voice"), \
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

    def test_mps_signature_tracks_locked_runtime_and_selected_source(self):
        from music_cover.common.resources import signature, execution_python, upstream_code
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / 'configs/macos'
            locks.mkdir(parents=True)
            lock = locks / 'yue2-torch215.txt'
            lock.write_text('torch==2.15.0')
            environment = root / 'envs/yue2'
            environment.parent.mkdir()
            environment.symlink_to(root / 'shared-yue-runtime')
            self.assertEqual(execution_python(root, 'yue2', 'cover', 'mps'), environment / 'bin/python')
            yue = root / 'third_party/YuE/model.py'
            seed = root / 'third_party/seed-vc/model.py'
            for source in (yue, seed):
                source.parent.mkdir(parents=True)
                source.write_text('original')
            process_env = {'PYTHONPATH': str(root / 'src')}
            with patch('music_cover.common.resources.resource_data', return_value={}), \
                 patch('music_cover.common.resources.subprocess.check_output',
                       return_value='{"packages":{"torch":"2.15.0"}}') as probe:
                original = signature(root, ['yue'], 'mps', environment=process_env)
                self.assertEqual(probe.call_args.kwargs['cwd'], root)
                self.assertEqual(probe.call_args.kwargs['env'], process_env)
                self.assertGreater(probe.call_args.kwargs['timeout'], 0)
                self.assertIn('runtime_identity()', probe.call_args.args[0][-1])
                identity = upstream_code(root, 'yue')
                seed.write_text('patched')
                self.assertEqual(identity, upstream_code(root, 'yue'))
                self.assertEqual(original, signature(root, ['yue'], 'mps'))
                lock.write_text('torch==2.15.0\n# updated hashes')
                updated = signature(root, ['yue'], 'mps')
                self.assertNotEqual(original, updated)
                yue.write_text('patched')
                self.assertNotEqual(updated, signature(root, ['yue'], 'mps'))
                self.assertNotEqual(identity, upstream_code(root, 'yue'))
                probe.return_value = '{"packages":{"torch":"2.8.0"}}'
                with self.assertRaisesRegex(ValueError, 'expected 2.15.0'):
                    signature(root, ['yue'], 'mps')
                probe.side_effect = subprocess.TimeoutExpired('runtime_identity', 90)
                with self.assertRaisesRegex(RuntimeError, 'probe timed out'):
                    signature(root, ['yue'], 'mps')

    def test_cuda_preflight_and_probe_receive_the_requested_mask(self):
        from music_cover import pipeline
        command = ['pipeline', '--demo', '--gpu', '2', '--stage', '01', '--preflight']
        with patch.object(sys, 'argv', command), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
             patch('music_cover.runtime.subprocess.check_output', return_value='1\n') as probe, \
             patch.object(pipeline, 'check_resources') as check:
            pipeline.main()
            self.assertEqual(probe.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            self.assertEqual(check.call_args.kwargs['environment']['CUDA_VISIBLE_DEVICES'], '2')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')

    def test_mps_stage_boundary_is_checked_without_probing_hardware(self):
        from music_cover import pipeline
        with patch.object(sys, 'argv', ['pipeline', '--stage', '08', '--device', 'mps', '--preflight']), \
             patch.object(pipeline, 'resolve_device') as probe, self.assertRaises(SystemExit) as error:
            pipeline.main()
        self.assertEqual(error.exception.code, 2)
        probe.assert_not_called()

    def test_backend_request_does_not_ignore_explicit_mps_options(self):
        from music_cover.backends.cli import main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            argv = ['backend', 'review_stems', '--device', 'mps', '--memory-budget-gib', '4', '--output', str(output)]
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_saved_cuda_options_when_resuming_with_auto(self):
        from music_cover import pipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input'
            source.touch()
            request = dict(song=str(source), lyrics=str(source), reference=str(source), style='folk',
                           gpu='2', memory_budget_gib=48)
            (root / 'request.json').write_text(json.dumps(request))
            (root / 'stage_state.json').write_text(json.dumps({'schema_version': 2}))
            (root / 'config.resolved.json').write_text('{}')
            resume = ['pipeline', '--resume', directory, '--device', 'auto', '--stage', '01']
            cases = [('cuda', [], '2', 48), ('cuda', ['--gpu', '1'], '1', 48),
                     ('mps', [], None, 16), ('mps', ['--memory-budget-gib', '12'], None, 12)]
            for actual, extra, gpu, budget in cases:
                with self.subTest(actual=actual, extra=extra), patch.object(sys, 'argv', [*resume, *extra]), \
                     patch.object(pipeline, 'resolve_device', return_value=actual), patch.object(pipeline, 'Pipeline') as runner:
                    pipeline.main()
                    selected = runner.call_args.args[1]
                    self.assertEqual((selected['device'], selected['gpu'], selected['memory_budget_gib']), (actual, gpu, budget))
            with patch.object(sys, 'argv', ['pipeline', '--resume', directory, '--stage', '06', '--preflight']), \
                 patch.object(pipeline, 'resolve_device') as resolve, patch.object(pipeline, 'check_resources') as check:
                pipeline.main()
                resolve.assert_not_called()
                self.assertEqual(check.call_args.kwargs['device'], 'cuda')
                self.assertTrue(check.call_args.kwargs['mp3'])


if __name__ == "__main__":
    unittest.main()
