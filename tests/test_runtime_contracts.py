"""Regression checks for shared profile integrity and interpreter resolution."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from music_cover.audio import sha256
from music_cover.common.resources import python_for, preflight
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
        for backend in ['rvc', 'so_vits_svc']:
            self.assertEqual(python_for('/project', backend), Path('/opt/conda/bin/python'))

    def test_metrics_stay_in_seed_environment(self):
        self.assertEqual(python_for('/project', 'voice_metrics'), Path('/project/envs/seed-vc/bin/python'))

    def test_missing_ffprobe_is_caught_before_processing(self):
        with patch('music_cover.common.resources.shutil.which', side_effect=lambda n: '/bin/ffmpeg' if n == 'ffmpeg' else None):
            with self.assertRaisesRegex(RuntimeError, 'ffprobe'):
                preflight('/project')


if __name__ == '__main__':
    unittest.main()
