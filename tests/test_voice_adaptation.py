"""Safety/compatibility contracts; real GPU verification is recorded separately."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import soundfile as sf
from music_cover.audio import sha256
from music_cover.common.io import write_json
from music_cover.voice import resolve_voice, BASE_CHECKPOINT, BASE_CONFIG
from music_cover import voice_training as training
from music_cover.modules import voice_conversion, evaluation


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {'preprocess_params': {'sr': 44100}, 'model_params': {'DiT': {'f0_condition': True}}}
        for relative, content in [(BASE_CHECKPOINT, b'base'), (BASE_CONFIG, json.dumps(self.config).encode())]:
            p = self.root / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(content)
    def tearDown(self): self.temp.cleanup()
    def profile(self):
        p = self.root / 'voice'; p.mkdir()
        (p / 'model.pth').write_bytes(b'trained')
        write_json(p / 'config.yml', self.config)
        write_json(p / 'validation.json', {'quality': 'integration_only'})
        data = {'schema_version': 1, 'backend': 'seed_vc', 'voice_id': 'example', 'version': 'v001'}
        for role, file in [('checkpoint','model.pth'),('config','config.yml'),('validation','validation.json')]:
            data[role] = {'path': file, 'sha256': sha256(p / file)}
        write_json(p / 'voice.json', data)
        return p / 'voice.json'
    def test_zero_shot_needs_no_training_assets_or_profile(self):
        result = resolve_voice('zero_shot', 'missing/profile.json', root=self.root)
        self.assertEqual(result['checkpoint'], self.root / BASE_CHECKPOINT)
        self.assertNotIn('profile', result)
    def test_finetuned_requires_explicit_profile_without_fallback(self):
        with self.assertRaisesRegex(ValueError, 'requires --voice-profile'):
            resolve_voice('finetuned', root=self.root)
    def test_profile_uses_selected_checkpoint(self):
        result = resolve_voice('finetuned', self.profile(), root=self.root)
        self.assertEqual(result['checkpoint'].read_bytes(), b'trained')
        self.assertEqual(result['version'], 'v001')
    def test_tampered_weights_rejected(self):
        profile = self.profile(); (profile.parent / 'model.pth').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            resolve_voice('finetuned', profile, root=self.root)
    def test_tampered_validation_rejected(self):
        profile = self.profile(); (profile.parent / 'validation.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            resolve_voice('finetuned', profile, root=self.root)
    def test_wrong_sample_rate_rejected(self):
        self.config['preprocess_params']['sr'] = 22050
        write_json(self.root / BASE_CONFIG, self.config)
        with self.assertRaisesRegex(ValueError, '44.1'):
            resolve_voice(root=self.root)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.root_patch = patch.object(training, 'ROOT', self.root); self.root_patch.start()
        self.out = self.root / 'output'; self.out.mkdir()
        t = np.arange(44100 * 2) / 44100
        sf.write(self.root / 'voice.wav', .2 * np.sin(2*np.pi*220*t), 44100)
        sf.write(self.root / 'other.wav', .2 * np.sin(2*np.pi*330*t), 44100)
        self.c = {'voice_id':'example','version':'v1','dataset_dir':'dataset','reference':'voice.wav',
                  'smoke_test':True,'segment_seconds':1,
                  'sources':[{'path':'voice.wav','group':'recording1','split':'train'}],
                  'training':{'max_steps':2,'save_every':1,'batch_size':1,'learning_rate':'1e-5'}, 'validation':{}}
    def tearDown(self): self.root_patch.stop(); self.temp.cleanup()
    def prepare(self): training.prepare(self.c, self.root, self.out, SimpleNamespace())
    def test_learning_rate_scientific_notation_is_numeric(self):
        write_json(self.root / 'config.yaml', self.c)
        self.assertEqual(training.load_config('config.yaml')['training']['learning_rate'], 1e-5)
    def test_same_recording_across_splits_rejected(self):
        self.c['sources'].append({'path':'voice.wav','group':'different_name','split':'validation'})
        with self.assertRaisesRegex(ValueError, 'leakage'): self.prepare()
    def test_same_group_across_splits_rejected(self):
        self.c['sources'].append({'path':'other.wav','group':'recording1','split':'validation'})
        with self.assertRaisesRegex(ValueError, 'leakage'): self.prepare()
    def test_formal_training_requires_holdouts(self):
        self.c['smoke_test'] = False
        with self.assertRaisesRegex(ValueError, 'validation/test'): self.prepare()
    def test_real_audio_preparation_and_corruption_detection(self):
        self.prepare(); m = training.dataset_manifest(self.c)
        self.assertEqual(len(m['clips']), 2)
        self.assertEqual(m['assessment'], 'integration_only')
        (self.root / 'dataset' / m['clips'][0]['path']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'clip changed'): training.dataset_manifest(self.c)
    def test_smoke_dataset_cannot_be_relabelled_formal(self):
        self.prepare(); self.c['smoke_test'] = False
        with self.assertRaisesRegex(ValueError, 'smoke status'): training.dataset_manifest(self.c)
    def test_changed_reference_requires_new_preparation(self):
        self.prepare(); self.c['reference'] = 'other.wav'
        with self.assertRaisesRegex(ValueError, 'Reference changed'): training.dataset_manifest(self.c)

    def validated_export(self, eligible=True):
        self.prepare()
        checkpoint = self.root / 'candidate.pth'; checkpoint.write_bytes(b'candidate')
        folder = self.root / 'validation'; folder.mkdir()
        write_json(folder / 'validation.json', {
            'dataset_identity': training.dataset_manifest(self.c)['identity'],
            'selected_checkpoint': str(checkpoint),
            'candidates': [{'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'eligible': eligible}]})
        write_json(self.root / 'state.json', {'stages': {'02_validate': {'output': str(folder)}}})
        return checkpoint

    def test_export_rejects_weights_changed_after_validation(self):
        checkpoint = self.validated_export(); checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'not validated or has changed'):
            training.export(self.c, self.root, self.out, SimpleNamespace(checkpoint=None))

    def test_export_rejects_failed_pitch_validation(self):
        self.validated_export(eligible=False)
        with self.assertRaisesRegex(ValueError, 'failed technical validation'):
            training.export(self.c, self.root, self.out, SimpleNamespace(checkpoint=None))

    def test_export_cannot_overwrite_a_voice_version(self):
        self.validated_export(); self.c['export_dir'] = 'published'
        (self.root / 'published').mkdir()
        (self.root / 'published/model.pth').write_bytes(b'keep this model')
        with self.assertRaisesRegex(ValueError, 'immutable'):
            training.export(self.c, self.root, self.out, SimpleNamespace(checkpoint=None))
        self.assertEqual((self.root / 'published/model.pth').read_bytes(), b'keep this model')


class InterfaceTests(unittest.TestCase):
    def test_default_conversion_keeps_zero_shot_and_no_pitch_shift(self):
        p = voice_conversion.params({'steps':30,'seed':42}, voice_conversion.DEFAULTS)
        self.assertEqual(p['mode'], 'zero_shot')
        self.assertEqual(p['semitones'], 0)
        self.assertFalse(p['auto_f0'])
    def test_voice_evaluation_is_independent_of_transposition(self):
        request = {'transpose':'0','review':True}
        c = {**evaluation.DEFAULTS, 'voice_check':True}
        self.assertIn('voice_metrics', evaluation.backends(request, c))
        p = evaluation.params(request, c)
        self.assertTrue(p['voice_check']); self.assertFalse(p['pitch_check'])
        self.assertNotIn('voice_metrics', evaluation.backends(request, evaluation.DEFAULTS))

    def test_ddp_weight_prefix_and_missing_parameter_guard(self):
        from music_cover.voice_weights import check_parameter_coverage
        value = np.zeros((2, 3))
        model = {'cfm': SimpleNamespace(named_parameters=lambda: [('weight', value)])}
        check_parameter_coverage(model, {'cfm': {'module.weight': value}})
        check_parameter_coverage(model, {'cfm': {'weight': value}})
        with self.assertRaisesRegex(ValueError, 'cfm.weight'):
            check_parameter_coverage(model, {'cfm': {'weight': np.zeros((3, 2))}})
        with self.assertRaisesRegex(ValueError, 'cfm.weight'):
            check_parameter_coverage(model, {})


if __name__ == '__main__': unittest.main()
