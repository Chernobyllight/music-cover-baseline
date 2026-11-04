"""SVC command-line configuration contracts; no model training or GPU required."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import yaml
from music_cover import svc_config as tool, svc_training, voice_data_config, voice_training
from music_cover.audio import sha256
from music_cover.common.io import write_json
from music_cover.common.voice_config import write_new_config
from music_cover.modules import voice_conversion
from music_cover.svc import UPSTREAM_COMMIT
from test_voice_training_config import fixture


class TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        fixture(self.root)
        shutil.rmtree(self.root / 'models/Seed-VC')
        for module in (tool, svc_training, voice_data_config, voice_training):
            patcher = patch.object(module, 'ROOT', self.root)
            patcher.start(); self.addCleanup(patcher.stop)
        args = voice_data_config.parser().parse_args([
            '--voice-id', 'singer', '--version', 'manual', '--recordings-dir', 'data/new/recordings',
            '--source-run', 'runs/song', '--validation-ids', 'c', '--test-ids', 'd'])
        config, _, _ = voice_data_config.build_config(args)
        voice_training.prepare(config, self.root/'runs/data', self.root/'runs/data/out', None)
        self.dataset = self.root / config['dataset_dir']
        self.argv = ['training', '--voice-id', 'singer', '--version', 'svc_manual',
                     '--source-dataset', str(self.dataset.relative_to(self.root)), '--source-run', 'runs/song',
                     '--validation-windows', '0:1', '2:1']

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() if p.is_file() else None for p in self.root.rglob('*')}

    def manifest(self, edit):
        file = self.dataset/'manifest.json'
        data = json.loads(file.read_text()); edit(data); write_json(file, data)

    def build(self, extra=()):
        return tool.training_config(tool.parser().parse_args([*self.argv, *extra]))

    def test_generated_config_consumed_by_svc_without_other_model_assets(self):
        config, out, summary = self.build()
        self.assertEqual(config['training'], dict(max_steps=4000, save_every=1000, batch_size=6, learning_rate=.0001, seed=42))
        self.assertEqual(summary['clips'], {'train': 2, 'validation': 1, 'test': 1})
        self.assertIn('attempt_007', config['reference'])
        self.assertIn('attempt_003', config['configuration_generation']['checked_stage_artifacts']['vocals']['path'])
        self.assertNotIn(str(self.root), yaml.safe_dump(config))
        write_new_config(out, config)
        loaded = svc_training.load_config(out)
        self.assertEqual(len(svc_training.source_manifest(loaded)['clips']), 4)

    def test_plan_writes_nothing_and_publication_writes_only_yaml(self):
        before = self.snapshot()
        with contextlib.redirect_stdout(io.StringIO()): tool.main([*self.argv, '--plan'])
        self.assertEqual(before, self.snapshot())
        with contextlib.redirect_stdout(io.StringIO()): tool.main(self.argv)
        files = [p for p in self.snapshot().keys()-before.keys() if (self.root/p).is_file()]
        self.assertEqual(files, ['configs/voice_training/so_vits_svc/singer_svc_manual.yaml'])
        after = self.snapshot()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): tool.main(self.argv)
        self.assertEqual(after, self.snapshot())

    def test_reference_mismatch_and_dataset_leakage_are_rejected(self):
        self.manifest(lambda m: m.update(reference_sha256='wrong'))
        with self.assertRaisesRegex(ValueError, 'Reference differs'): self.build()
        self.manifest(lambda m: m.update(reference_sha256=sha256(self.root/'runs/song/stages/00_prepare/attempt_007/artifacts/reference.wav')))
        self.manifest(lambda m: m['clips'][-1].update(group=m['clips'][0]['group']))
        with self.assertRaisesRegex(ValueError, 'split leakage'): self.build()

    def test_changed_audio_is_rejected(self):
        clip = next(self.dataset.glob('train/*.wav'))
        with clip.open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Source dataset'): self.build()

    def test_wrong_rate_stereo_clipping_and_nonfinite_audio_rejected(self):
        manifest = json.loads((self.dataset/'manifest.json').read_text())
        row = manifest['clips'][0]; clip = self.dataset/row['path']
        original = clip.read_bytes()
        for rate, data in [(48000, np.zeros(96000)), (44100, np.zeros((88200, 2))),
                           (44100, np.ones(88200)), (44100, np.full(88200, np.nan))]:
            with self.subTest(rate=rate, shape=data.shape):
                sf.write(clip, data, rate, subtype='FLOAT')
                self.manifest(lambda m: m['clips'][0].update(sha256=sha256(clip)))
                with self.assertRaises(ValueError): self.build(['--peak-policy', 'error'])
                clip.write_bytes(original); write_json(self.dataset/'manifest.json', manifest)

    def test_invalid_windows_parameters_and_output_destinations_rejected(self):
        for extra in [('--validation-windows', '4:2'), ('--validation-windows', '0:2', '1:2'),
                      ('--max-steps', '0'), ('--learning-rate', 'nan'), ('--seed', '-1'),
                      ('--gpu', '-1'), ('--voice-id', 'a.b'), ('--version', '../bad'),
                      ('--peak-limit', 'nan'), ('--peak-limit', '1'),
                      ('--dataset-dir', str(self.dataset/'nested'))]:
            with self.subTest(extra=extra), self.assertRaises(ValueError): self.build(extra)

    def test_failed_or_changed_upstream_stage_rejected(self):
        state_file = self.root/'runs/song/stage_state.json'
        state = json.loads(state_file.read_text()); original = copy.deepcopy(state)
        state['stages']['04_vocal_separation']['status'] = 'failed'; write_json(state_file, state)
        with self.assertRaisesRegex(ValueError, 'successful'): self.build()
        write_json(state_file, original)
        vocals = self.root/'runs/song/stages/04_vocal_separation/attempt_003/artifacts/vocals.wav'
        with vocals.open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact missing or changed'): self.build()

    def test_config_generation_is_cwd_and_mount_independent(self):
        original, _, _ = self.build()
        with tempfile.TemporaryDirectory() as directory:
            moved = Path(directory)/'different-mount/project'
            shutil.copytree(self.root, moved)
            cwd = Path.cwd()
            try:
                os.chdir(directory)
                with patch.object(tool, 'ROOT', moved): actual, _, _ = self.build()
            finally: os.chdir(cwd)
        self.assertEqual(original, actual)

    def test_prepare_consumes_peak_policy_and_preserves_original_audio(self):
        manifest = json.loads((self.dataset/'manifest.json').read_text())
        clip = self.dataset/manifest['clips'][0]['path']
        audio, rate = sf.read(clip, dtype='float32')
        sf.write(clip, audio*20, rate, subtype='FLOAT')
        self.manifest(lambda m: m['clips'][0].update(sha256=sha256(clip)))
        before = clip.read_bytes()
        config, _, summary = self.build()
        self.assertEqual(len(summary['clips_requiring_attenuation']), 1)
        for name in ('svc_contentvec.py', 'svc_train.py'):
            file = self.root/'src/music_cover/backends'/name
            file.parent.mkdir(parents=True, exist_ok=True); file.write_text('fixture only')
        template = {'train': {}, 'data': {'sampling_rate': 44100, 'hop_length': 512},
                    'model': {'speech_encoder': 'vec768l12', 'ssl_dim': 768, 'gin_channels': 768,
                              'n_speakers': 1, 'vocoder_name': 'nsf-hifigan'}, 'spk': {}}
        write_json(self.root/'third_party/so-vits-svc/configs_template/config_template.json', template)
        run = self.root/'runs/svc'
        with patch.object(svc_training, 'check_assets', return_value={}), \
                patch('music_cover.backends.svc_train.prepare_features'):
            svc_training.prepare(config, run, run/'artifacts', None)
        prepared = self.root/config['dataset_dir']
        result = json.loads((prepared/'manifest.json').read_text())
        row = result['clips'][0]
        converted, _ = sf.read(prepared/row['path'])
        self.assertLess(np.max(np.abs(converted)), 1.)
        self.assertLess(row['audio_preparation']['gain'], 1.)
        self.assertEqual(row['source_sha256'], sha256(clip))
        self.assertEqual(clip.read_bytes(), before)


class InferenceConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.folder = self.root/'models/voices/singer/so_vits_svc/manual'
        self.folder.mkdir(parents=True)
        patcher = patch.object(tool, 'ROOT', self.root); patcher.start(); self.addCleanup(patcher.stop)
        config = {'data': {'sampling_rate': 44100, 'hop_length': 512},
                  'model': {'speech_encoder': 'vec768l12', 'ssl_dim': 768, 'gin_channels': 768,
                            'n_speakers': 1, 'vocoder_name': 'nsf-hifigan'}, 'spk': {'singer': 0}}
        (self.folder/'model.pth').write_bytes(b'fixture only, never loaded')
        write_json(self.folder/'config.json', config); write_json(self.folder/'validation.json', {})
        write_json(self.root/'configs/so_vits_svc_assets.lock.json', {'assets': []})
        self.profile = {'schema_version': 1, 'backend': 'so_vits_svc', 'adaptation_scope': 'speaker',
                        'voice_id': 'singer', 'version': 'manual', 'speaker_name': 'singer',
                        'upstream_commit': UPSTREAM_COMMIT, 'inference_assets': {}}
        for key, name in [('checkpoint', 'model.pth'), ('config', 'config.json'), ('validation', 'validation.json')]:
            self.profile[key] = {'path': name, 'sha256': sha256(self.folder/name)}
        self.save()
        self.argv = ['inference', '--voice-profile', str((self.folder/'voice.json').relative_to(self.root))]

    def save(self): write_json(self.folder/'voice.json', self.profile)

    def build(self, extra=()): return tool.inference_config(tool.parser().parse_args([*self.argv, *extra]))

    def test_config_matches_pipeline_and_explicit_settings_override_profile(self):
        self.profile['recommended_settings'] = {'noise_scale': .1, 'chunk_seconds': 8.}
        self.save(); config, _, _ = self.build(['--noise-scale', '.2', '--seed', '7'])
        module = config['modules']['05_voice_conversion']
        self.assertEqual(module['candidate_seeds'], [7]); self.assertEqual(module['candidate_count'], 1)
        self.assertEqual(module['svc']['noise_scale'], .2); self.assertEqual(module['svc']['chunk_seconds'], 8.)
        params = voice_conversion.params({'seed': 42, 'steps': 30}, {**voice_conversion.DEFAULTS, **module})
        self.assertEqual(params['backend'], 'so_vits_svc'); self.assertIsNone(params['steps'])
        self.assertNotIn(str(self.root), yaml.safe_dump(config))

    def test_plan_missing_model_and_no_overwrite(self):
        before = set(self.root.rglob('*'))
        with contextlib.redirect_stdout(io.StringIO()): tool.main([*self.argv, '--plan'])
        self.assertEqual(before, set(self.root.rglob('*')))
        with contextlib.redirect_stdout(io.StringIO()): tool.main(self.argv)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): tool.main(self.argv)
        (self.folder/'model.pth').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)): self.build(['--output', 'configs/new.yaml'])

    def test_wrong_backend_invalid_settings_and_seed_rejected(self):
        for extra in [('--f0-predictor', 'other'), ('--noise-scale', '2'), ('--seed', '-1'),
                      ('--overlap-seconds', '25'), ('--pad-seconds', '0')]:
            with self.subTest(extra=extra), self.assertRaises(ValueError): self.build(extra)
        self.profile['backend'] = 'rvc'; self.save()
        with self.assertRaisesRegex(ValueError, 'so_vits_svc profile'): self.build()


class EntryTests(unittest.TestCase):
    def test_svc_scripts_use_base_without_legacy_controller(self):
        root = Path(__file__).resolve().parents[1]/'scripts/05_voice_conversion/so_vits_svc'
        scripts = [root/'run.sh', root/'preflight/run.sh', root/'create_inference_config/run.sh',
                   *sorted((root/'training').glob('*/run.sh'))]
        self.assertEqual(len(scripts), 8)
        for file in scripts:
            text = file.read_text()
            self.assertIn('/opt/conda/bin/python', text)
            self.assertNotIn('/envs/', text); self.assertNotIn('run_baseline.sh', text)


class AudioPreparationTests(unittest.TestCase):
    def test_float_overrange_is_attenuated_without_changing_source_or_shape(self):
        source = np.array([[1.5], [-2.0], [0.2]], dtype=np.float32)
        before = source.copy()
        output, record = svc_training.prepare_audio(source, 44100, svc_training.audio_settings({
            'audio': {'peak_policy': 'attenuate_if_needed', 'peak_limit': .99}}))
        np.testing.assert_array_equal(source, before)
        np.testing.assert_allclose(output, source * (.99 / 2))
        self.assertAlmostEqual(record['gain'], .495)
        self.assertLess(np.max(np.abs(output)), 1)

    def test_legacy_policy_and_in_range_audio_unchanged(self):
        settings = svc_training.audio_settings({})
        with self.assertRaisesRegex(ValueError, 'needs attenuation'):
            svc_training.prepare_audio(np.array([[1.2]], dtype=np.float32), 44100, settings)
        source = np.array([[.2], [-.9]], dtype=np.float32)
        output, record = svc_training.prepare_audio(source, 44100, settings)
        np.testing.assert_array_equal(source, output)
        self.assertEqual(record['gain'], 1.)


if __name__ == '__main__':
    unittest.main()
