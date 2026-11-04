import json
import tempfile
import unittest
from pathlib import Path
import yaml
from music_cover.common.voice_paths import runtime_model_config, training_output


class PortableConfigTests(unittest.TestCase):
    def test_export_config_resolves_under_new_root_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('vocoder', 'tokenizer'):
                (root / 'models' / name).mkdir(parents=True)
            for legacy in (False, True):
                prefix = '/retired/mount/project/' if legacy else ''
                cfg = {'log_dir': prefix + 'runs/train/upstream', 'model_params': {
                    'vocoder': {'name': prefix + 'models/vocoder'},
                    'speech_tokenizer': {'name': prefix + 'models/tokenizer'}}}
                source = root / 'export.yml'; source.write_text(yaml.safe_dump(cfg))
                before = source.read_bytes()
                target = runtime_model_config(source, root / 'runtime.yml', root)
                actual = yaml.safe_load(target.read_text())
                self.assertEqual(actual['model_params']['vocoder']['name'], str(root / 'models/vocoder'))
                self.assertEqual(source.read_bytes(), before)

    def test_unknown_missing_absolute_asset_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'export.yml'
            source.write_text(yaml.safe_dump({'log_dir': 'runs/test', 'model_params': {
                'vocoder': {'name': '/unknown/custom-vocoder'},
                'speech_tokenizer': {'name': 'models/tokenizer'}}}))
            with self.assertRaises(FileNotFoundError):
                runtime_model_config(source, root / 'runtime.yml', root)

    def test_training_state_relocates_exact_stage_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for old in ('stages/01_train/attempt_002/artifacts',
                        '/retired/project/runs/train/stages/01_train/attempt_002/artifacts'):
                (run / 'state.json').write_text(json.dumps({'stages': {'01_train': {'output': old}}}))
                self.assertEqual(training_output(run, '01_train'), run / 'stages/01_train/attempt_002/artifacts')
