import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml
from music_cover import voice_data_config as data_tool, rvc_config as rvc_tool, voice_training
from music_cover.common.voice_config import write_new_config
from test_voice_training_config import fixture
import test_rvc


class ConfigTools(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); fixture(self.root)
        # Dataset and RVC config generation must not depend on Seed-VC assets.
        shutil.rmtree(self.root/'models/Seed-VC')
        for module in (data_tool, rvc_tool, voice_training):
            patcher = patch.object(module, 'ROOT', self.root); patcher.start(); self.addCleanup(patcher.stop)
        self.data_argv = ['--voice-id', 'singer', '--version', 'manual', '--recordings-dir', 'data/new/recordings',
                          '--source-run', 'runs/song', '--validation-ids', 'c', '--test-ids', 'd']
        self.data_args = data_tool.parser().parse_args(self.data_argv)

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() if p.is_file() else None for p in self.root.rglob('*')}

    def prepare(self):
        c, out, _ = data_tool.build_config(self.data_args); write_new_config(out, c)
        loaded = voice_training.load_config(out)
        voice_training.prepare(loaded, self.root/'runs/data', self.root/'runs/data/out', None)
        return c, out

    def train_args(self, extra=()):
        return rvc_tool.parser().parse_args(['training', '--voice-id', 'singer', '--version', 'rvc_manual',
            '--source-dataset', 'data/voice_training/singer/manual/dataset', '--source-run', 'runs/song',
            '--validation-windows', '0:1', '2:1', *extra])

    def test_data_config_is_model_independent_and_prepares_real_audio(self):
        c, _ = self.prepare()
        self.assertNotIn('base_checkpoint', c); self.assertNotIn('training', c)
        m = json.loads((self.root/c['dataset_dir']/'manifest.json').read_text())
        self.assertEqual({r['split'] for r in m['clips']}, {'train', 'validation', 'test'})
        self.assertEqual(len(m['clips']), 4)

    def test_data_plan_and_publish_write_only_the_requested_yaml(self):
        before = self.snapshot()
        with contextlib.redirect_stdout(io.StringIO()): data_tool.main([*self.data_argv, '--plan'])
        self.assertEqual(before, self.snapshot())
        with contextlib.redirect_stdout(io.StringIO()): data_tool.main(self.data_argv)
        created = [p for p in self.snapshot().keys()-before.keys() if (self.root/p).is_file()]
        self.assertEqual(created, ['configs/voice_data/singer_manual.yaml'])
        published = self.snapshot()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): data_tool.main(self.data_argv)
        self.assertEqual(published, self.snapshot())

    def test_data_split_and_target_validation(self):
        for key, value in [('validation_ids', ['c','c']), ('test_ids', ['c']), ('segment_seconds', float('nan')),
                           ('voice_id', '../bad'), ('gpu', -1), ('dataset_dir', 'data/new/recordings/nested')]:
            a = data_tool.parser().parse_args(self.data_argv); setattr(a, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError): data_tool.build_config(a)

    def test_dataset_config_cannot_accidentally_train_seed_vc(self):
        _, out = self.prepare(); before = self.snapshot()
        argv = ['training', '--config', str(out), '--run', str(self.root/'runs/do-not-create'), '--stage', '01_train']
        with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            voice_training.main()
        self.assertEqual(before, self.snapshot())

    def test_training_config_consumes_new_dataset_and_current_stage_attempt(self):
        self.prepare(); c, out, summary = rvc_tool.training_config(self.train_args())
        self.assertEqual(c['training']['max_steps'], 4000)
        self.assertEqual(c['inference']['index_rate'], .5)
        self.assertEqual(summary['clips'], {'train': 2, 'validation': 1, 'test': 1})
        self.assertEqual(c['validation']['cases'][1], {'source_run': 'runs/song', 'start': 2., 'duration': 1.})
        self.assertIn('attempt_003', c['configuration_generation']['checked_stage_artifacts']['vocals']['path'])
        self.assertNotIn(str(self.root), yaml.safe_dump(c))
        from music_cover import rvc_training
        with patch.object(rvc_training, 'ROOT', self.root):
            self.assertEqual(len(rvc_training.source_manifest(c)['clips']), 4)

    def test_training_plan_no_side_effects_and_publish_no_overwrite(self):
        self.prepare(); a = self.train_args(); argv = ['training', '--voice-id', a.voice_id, '--version', a.version,
            '--source-dataset', a.source_dataset, '--source-run', a.source_run, '--validation-windows', '0:1']
        before = self.snapshot()
        with contextlib.redirect_stdout(io.StringIO()): rvc_tool.main([*argv, '--plan'])
        self.assertEqual(before, self.snapshot())
        with contextlib.redirect_stdout(io.StringIO()): rvc_tool.main(argv)
        after = self.snapshot()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): rvc_tool.main(argv)
        self.assertEqual(after, self.snapshot())

    def test_training_rejects_invalid_parameters_windows_and_destinations(self):
        self.prepare()
        for extra in [('--max-steps','0'), ('--learning-rate','nan'), ('--seed','-1'), ('--gpu','-1'),
                      ('--index-rate','1.1'), ('--input-crop-frames','0'), ('--overlap-seconds','8'),
                      ('--validation-windows','4:2'), ('--validation-windows','0:2','1:2'),
                      ('--dataset-dir','data/voice_training/singer/manual/dataset'),
                      ('--output','new/config.yaml','--export-dir','new')]:
            with self.subTest(extra=extra), self.assertRaises(ValueError): rvc_tool.training_config(self.train_args(extra))

    def test_changed_audio_and_split_leakage_rejected(self):
        c, _ = self.prepare(); folder = self.root/c['dataset_dir']; path = folder/'manifest.json'
        m = json.loads(path.read_text()); old = path.read_bytes()
        m['clips'][-1]['group'] = m['clips'][0]['group']; path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, 'leakage'): rvc_tool.training_config(self.train_args())
        path.write_bytes(old); f = folder/m['clips'][0]['path']; f.write_bytes(f.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'): rvc_tool.training_config(self.train_args())

    def test_project_move_and_foreign_cwd_preserve_configuration(self):
        self.prepare(); original = rvc_tool.training_config(self.train_args())[0]
        moved = self.root/'moved'; content = self.root/'project-copy'
        # Copy only task inputs, preserving project-relative stage metadata.
        content.mkdir()
        for directory in ('data','runs'): shutil.copytree(self.root/directory, content/directory)
        content.rename(moved)
        old = Path.cwd()
        try:
            os.chdir('/')
            with patch.object(rvc_tool, 'ROOT', moved): actual = rvc_tool.training_config(self.train_args())[0]
        finally: os.chdir(old)
        self.assertEqual(original, actual)


class InferenceConfig(unittest.TestCase):
    def setUp(self):
        self.case = test_rvc.Profiles(); self.case.setUp(); self.addCleanup(self.case.tearDown)
        self.root = self.case.root
        self.case.profile['recommended_settings'] = {'index_rate': .75, 'protect': .4}
        self.case.save()
        patcher = patch.object(rvc_tool, 'ROOT', self.root); patcher.start(); self.addCleanup(patcher.stop)
        self.argv = ['inference', '--voice-profile', 'voice/voice.json']

    def test_profile_defaults_explicit_overrides_and_pipeline_contract(self):
        a = rvc_tool.parser().parse_args(self.argv)
        c, _, _ = rvc_tool.inference_config(a); m = c['modules']['05_voice_conversion']
        self.assertEqual(m['rvc']['index_rate'], .75); self.assertEqual(m['rvc']['protect'], .4)
        self.assertEqual(m['candidate_seeds'], [42]); self.assertEqual(m['candidate_count'], 1)
        from music_cover.modules.voice_conversion import params, DEFAULTS
        self.assertEqual(params({'seed': 42, 'steps': 30}, {**DEFAULTS, **m})['backend'], 'rvc')
        a = rvc_tool.parser().parse_args([*self.argv, '--index-rate', '.5', '--candidate-count', '3', '--seed', '10'])
        m = rvc_tool.inference_config(a)[0]['modules']['05_voice_conversion']
        self.assertEqual(m['rvc']['index_rate'], .5); self.assertEqual(m['candidate_seeds'], [10,11,12])

    def test_inference_plan_is_read_only_and_existing_output_is_preserved(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()): rvc_tool.main([*self.argv, '--plan'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        with contextlib.redirect_stdout(io.StringIO()): rvc_tool.main(self.argv)
        output = self.root/'configs/voice_inference/rvc/huang_qi_shan_test.yaml'; old = output.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): rvc_tool.main(self.argv)
        self.assertEqual(old, output.read_bytes())

    def test_corrupt_model_and_invalid_candidates_rejected(self):
        for extra in [('--candidate-count','0'), ('--seed','-1'), ('--protect','.6'),
                      ('--candidate-count','2','--candidate-seeds','1','1'), ('--output','voice/new.yaml')]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                rvc_tool.inference_config(rvc_tool.parser().parse_args([*self.argv, *extra]))
        (self.root/'voice/model.pth').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            rvc_tool.inference_config(rvc_tool.parser().parse_args(self.argv))


class Preflight(unittest.TestCase):
    def test_checks_runtime_scope_and_hashes_without_writing(self):
        from music_cover import rvc_preflight as check
        from music_cover.audio import sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'configs').mkdir(); asset = root/'embedding'; asset.write_bytes(b'weights')
            (root/'configs/voice_training_assets.lock.json').write_text(json.dumps({'assets':[
                {'scope':'evaluation','path':'embedding','sha256':sha256(asset)}]}))
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda n: 'GPU'))
            a = check.parser().parse_args(['--scope','inference'])
            with patch.object(check,'ROOT',root), patch.dict('sys.modules',{'torch':torch}), \
                 patch.object(check,'preflight') as dependency, patch.object(check,'check_assets',return_value={'assets':[]}) as assets:
                result = check.check(a)
                self.assertFalse(result['writes']); assets.assert_called_with(root, training=False)
                dependency.assert_called_once_with(root, ['rvc','voice_metrics'], mp3=True)
                asset.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'corrupt'): check.check(a)

    def test_invalid_gpu_fails_before_runtime_import(self):
        from music_cover import rvc_preflight as check
        with self.assertRaisesRegex(ValueError,'nonnegative'):
            check.check(check.parser().parse_args(['--gpu','-1']))


if __name__ == '__main__': unittest.main()
