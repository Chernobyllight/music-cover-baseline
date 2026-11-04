"""Regression checks for shared profile integrity and interpreter resolution."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from music_cover.audio import sha256
from music_cover.common.resources import python_for, preflight, signature
from music_cover.engine import Pipeline
from music_cover.common.voice_profiles import verified_files
from music_cover.voice import merge_voice_config
from music_cover.modules.voice_conversion import DEFAULTS, params


class BackendSwitching(unittest.TestCase):
    def test_rvc_to_svc_clears_previous_index_and_seeds(self):
        old = {**DEFAULTS, 'backend': 'rvc', 'mode': 'finetuned', 'voice_profile': 'rvc.json',
               'candidate_seeds': [42, 43, 44], 'rvc': {'index_rate': .75}}
        new = merge_voice_config(old, {'backend': 'so_vits_svc', 'voice_profile': 'svc.json', 'svc': {'noise_scale': .1}})
        validated = params({'seed': 42, 'steps': 30}, new)
        self.assertEqual(validated['candidate_seeds'], [42])
        self.assertEqual(new['rvc'], {})
        self.assertEqual(old['rvc']['index_rate'], .75)

    def test_svc_to_zero_shot_seed_keeps_explicit_mode(self):
        old = {**DEFAULTS, 'backend': 'so_vits_svc', 'mode': 'finetuned', 'voice_profile': 'svc.json', 'svc': {'noise_scale': .1}}
        new = merge_voice_config(old, {'backend': 'seed_vc', 'mode': 'zero_shot'})
        self.assertEqual(params({'seed': 42, 'steps': 30}, new)['mode'], 'zero_shot')
        self.assertIsNone(new['voice_profile'])

    def test_same_backend_retains_user_preset(self):
        old = {**DEFAULTS, 'backend': 'rvc', 'rvc': {'index_rate': .75}, 'candidate_seeds': [42, 43, 44]}
        self.assertEqual(merge_voice_config(old, {}), old)

    def test_explicit_incompatible_settings_are_still_rejected(self):
        new = merge_voice_config(DEFAULTS, {'backend': 'rvc', 'mode': 'finetuned', 'voice_profile': 'rvc.json', 'svc': {'noise_scale': .1}})
        with self.assertRaisesRegex(ValueError, 'require'):
            params({'seed': 42, 'steps': 30}, new)


class SharedProfileContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / 'voice'
        self.directory.mkdir()
        self.weight = self.directory / 'model.pth'
        self.weight.write_bytes(b'verified model')
        self.profile = {'checkpoint': {'path': 'model.pth', 'sha256': sha256(self.weight)}}

    def tearDown(self):
        self.temp.cleanup()

    def test_matches_exported_model(self):
        files = verified_files(self.directory / 'voice.json', self.profile, ['checkpoint'])
        self.assertEqual(files['checkpoint'], self.weight.resolve())

    def test_changed_weight_never_loads(self):
        self.weight.write_bytes(b'changed model')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            verified_files(self.directory / 'voice.json', self.profile, ['checkpoint'])

    def test_rvc_boundary_checks_resolved_symlink(self):
        outside = self.root / 'outside.pth'
        outside.write_bytes(self.weight.read_bytes())
        self.weight.unlink()
        self.weight.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'path mismatch'):
            verified_files(self.directory / 'voice.json', self.profile, ['checkpoint'], contained=True)

    def test_legacy_seed_and_svc_external_assets_remain_compatible(self):
        self.profile['checkpoint']['path'] = str(self.weight.resolve())
        self.assertEqual(verified_files(self.root / 'voice.json', self.profile, ['checkpoint'])['checkpoint'], self.weight)


class RuntimeContracts(unittest.TestCase):
    def test_base_backends_share_existing_interpreter(self):
        for backend in ['rvc', 'so_vits_svc', 'separator', 'rmvpe', 'yue', 'seed_vc', 'voice_metrics']:
            self.assertEqual(python_for('/project', backend), Path('/opt/conda/bin/python'))

    def test_actual_backend_commands_match_base_preflight_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = object.__new__(Pipeline)
            runner.root = root
            runner.request = {'gpu': '0'}
            runner.environment = {}
            commands = {'backend_commands': []}
            runner.current = (root, root / 'process.log', commands, {})
            with patch('music_cover.engine.subprocess.run') as run:
                run.return_value.returncode = 0
                for environment, operation, backend in [('separator', 'separate', 'separator'), ('seed-vc', 'pitch', 'rmvpe'),
                                                         ('yue', 'cover', 'yue'), ('yue2', 'cover', 'yue'),
                                                         ('seed-vc', 'convert', 'seed_vc'),
                                                         ('seed-vc', 'convert_candidates', 'voice_metrics'),
                                                         ('seed-vc', 'evaluate_voice', 'voice_metrics')]:
                    runner.execute(environment, operation, root / operation)
                    self.assertEqual(run.call_args.args[0][0], str(python_for(root, backend)))
                    record = json.loads((root / 'command.json').read_text())['backend_commands'][-1]
                    self.assertEqual(record['argv'][0], '/opt/conda/bin/python')
                    self.assertEqual(record['environment'], '/opt/conda')

    def test_other_operations_keep_their_existing_interpreters(self):
        from music_cover.common.resources import execution_python
        for environment, operation in [('separator', 'review_stems'), ('separator', 'review_midi'),
                                       ('sheetsage2', 'transcribe')]:
            self.assertEqual(execution_python('/project', environment, operation),
                             Path('/project/envs') / environment / 'bin/python')

    def test_base_pitch_signature_tracks_its_lock_under_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            lock = root / 'configs/pitch-base.installed.txt'
            lock.write_text('audio-separator==0.30.2\n')
            with patch('music_cover.common.resources.resource_data', return_value={}):
                first = signature(root, ['separator', 'rmvpe'])
                for backend in first:
                    self.assertEqual(first[backend]['environment'], '/opt/conda')
                    self.assertEqual(first[backend]['installed_lock_sha256'], sha256(lock))
                lock.write_text('audio-separator==changed\n')
                self.assertNotEqual(first, signature(root, ['separator', 'rmvpe']))

    def test_missing_ffprobe_is_caught_before_processing(self):
        with patch('music_cover.common.resources.shutil.which', side_effect=lambda n: '/bin/ffmpeg' if n == 'ffmpeg' else None):
            with self.assertRaisesRegex(RuntimeError, 'ffprobe'):
                preflight('/project')

    def test_base_yue_signature_tracks_its_own_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            lock = root / 'configs/yue-base.installed.txt'
            lock.write_text('yue2-infer==0.1.6\n')
            with patch('music_cover.common.resources.resource_data', return_value={}):
                first = signature(root, ['yue'])
                self.assertEqual(first['yue']['environment'], '/opt/conda')
                self.assertEqual(first['yue']['installed_lock_sha256'], sha256(lock))
                (root / 'configs/yue2.installed.txt').write_text('legacy environment changed\n')
                self.assertEqual(first, signature(root, ['yue']))
                lock.write_text('yue2-infer==changed\n')
                self.assertNotEqual(first, signature(root, ['yue']))

    def test_seed_and_metrics_cache_track_base_not_legacy_venv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            lock = root / 'configs/seed-vc-base.installed.txt'
            lock.write_text('munch==4.0.0\n')
            (root / 'configs/seed_svc.local.yml').write_text('local model config\n')
            (root / 'configs/requirements-voice-adaptation.txt').write_text('speechbrain==1.0.3\n')
            with patch('music_cover.common.resources.resource_data', return_value={}):
                first = signature(root, ['seed_vc', 'voice_metrics'])
                for backend in first:
                    self.assertEqual(first[backend]['environment'], '/opt/conda')
                    self.assertEqual(first[backend]['installed_lock_sha256'], sha256(lock))
                (root / 'configs/seed-vc.installed.txt').write_text('legacy environment changed\n')
                self.assertEqual(first, signature(root, ['seed_vc', 'voice_metrics']))
                lock.write_text('munch==changed\n')
                self.assertNotEqual(first, signature(root, ['seed_vc', 'voice_metrics']))


if __name__ == '__main__':
    unittest.main()
