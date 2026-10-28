"""Controller integration tests use real files and manifests, without GPU mocks passing as inference."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from music_cover.engine import Pipeline, DependencyError
from music_cover.modules import load_modules, active_ids
from music_cover.pipeline import select_stages
from music_cover.common.paths import ROOT


class ModularExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'src/music_cover/common').mkdir(parents=True)
        shutil.copyfile(ROOT / 'src/music_cover/common/io.py', self.root / 'src/music_cover/common/io.py')
        self.input = self.root / 'input.txt'
        self.input.write_text('original')
        self.request = {'gpu': '0', 'review': False, 'variant': 'one'}
        self.calls = []
        self.modules = []
        for number, name in enumerate(['00_prepare', '01_source_score', '02_pitch_adaptation']):
            path = self.root / (name + '.py')
            path.write_text('# initial implementation\n')
            previous = None if number == 0 else self.modules[-1].ID
            def inputs(ctx, previous=previous):
                return {'input': ctx.output(previous, 'text') if previous else self.input}
            def action(ctx, out, inputs, params, name=name):
                self.calls.append(name)
                target = out / 'output.txt'
                target.write_text(inputs['input'].read_text() + params.get('variant', ''))
                return {'text': target}
            self.modules.append(SimpleNamespace(ID=name, __file__=str(path), CODE=[], DEFAULTS={},
                requires=lambda r, previous=previous: [previous] if previous else [], inputs=inputs,
                params=lambda r,c,number=number: {'variant':r['variant']} if number == 2 else {},
                backends=lambda r,c: [], run=action))
        self.preflight = patch('music_cover.engine.resources.preflight')
        self.preflight.start()
    def tearDown(self):
        self.preflight.stop()
        self.temp.cleanup()
    def pipeline(self, request=None):
        return Pipeline(self.root / 'run', request or self.request, root=self.root, modules=self.modules)
    def run_stages(self, selected=None, request=None, force=False):
        with self.pipeline(request) as p:
            p.run_all(selected or [m.ID for m in self.modules], force=force)
            return json.loads(p.state_path.read_text())

    def test_complete_stage_contract_and_resume_without_execution(self):
        first = self.run_stages()
        self.assertEqual(len(self.calls), 3)
        second = self.run_stages()
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(all(e['action'] == 'reused' for e in second['last_execution']))
        for record in first['stages'].values():
            attempt = Path(record['attempt'])
            for name in ['inputs.json','config.json','command.json','process.log','checks.json','outputs.json']:
                self.assertTrue((attempt / name).is_file(), name)
            output = json.loads((attempt / 'outputs.json').read_text())
            self.assertEqual(output['status'], 'succeeded')
            self.assertFalse(Path(output['artifacts']['text']['path']).is_absolute())
        self.assertEqual(second['status'], 'partial')
        self.assertFalse((self.root / 'run/deliverables').exists())

    def test_missing_dependency_is_not_implicitly_run(self):
        with self.assertRaisesRegex(DependencyError, '00_prepare'):
            self.run_stages(['02_pitch_adaptation'])
        self.assertEqual(self.calls, [])

    def test_corrupt_dependency_blocks_single_stage_and_is_rebuilt_on_resume(self):
        state = self.run_stages()
        Path(state['stages']['01_source_score']['outputs']['text']['path']).write_text('corrupt')
        with self.assertRaisesRegex(DependencyError, '01_source_score'):
            self.run_stages(['02_pitch_adaptation'])
        self.run_stages()
        self.assertEqual(self.calls.count('01_source_score'), 2)
        self.assertEqual(self.calls.count('00_prepare'), 1)
        # Rebuilt dependency has identical bytes, so the downstream artifact remains usable.
        self.assertEqual(self.calls.count('02_pitch_adaptation'), 1)

    def test_parameter_change_only_reruns_affected_stage(self):
        self.run_stages()
        self.run_stages(request={**self.request, 'variant': 'two'})
        self.assertEqual(self.calls, ['00_prepare','01_source_score','02_pitch_adaptation','02_pitch_adaptation'])

    def test_unrelated_module_edit_does_not_invalidate_earlier_modules(self):
        self.run_stages()
        Path(self.modules[2].__file__).write_text('# changed last module\n')
        self.run_stages()
        self.assertEqual(self.calls.count('00_prepare'), 1)
        self.assertEqual(self.calls.count('01_source_score'), 1)
        self.assertEqual(self.calls.count('02_pitch_adaptation'), 2)

    def test_contract_change_is_scoped_to_one_module(self):
        self.run_stages()
        self.modules[2].CONTRACT_VERSION = 2
        state = self.run_stages()
        self.assertEqual(self.calls.count('00_prepare'), 1)
        self.assertEqual(self.calls.count('01_source_score'), 1)
        self.assertEqual(self.calls.count('02_pitch_adaptation'), 2)
        self.assertEqual(state['stages']['02_pitch_adaptation']['contract_version'], 2)

    def test_failure_keeps_previous_outputs_and_retries_in_new_attempt(self):
        state = self.run_stages()
        original = Path(state['stages']['02_pitch_adaptation']['outputs']['text']['path'])
        action = self.modules[2].run
        self.modules[2].run = lambda ctx,out,i,p: {'missing': out / 'missing.wav'}
        with self.assertRaisesRegex(RuntimeError, 'Missing stage artifact'):
            self.run_stages(['02_pitch_adaptation'], force=True)
        state = json.loads((self.root / 'run/stage_state.json').read_text())
        failure = Path(state['stages']['02_pitch_adaptation']['attempt'])
        self.assertFalse((failure / 'outputs.json').exists())
        self.assertTrue((failure / 'failure.json').exists())
        self.assertTrue(original.is_file())
        self.modules[2].run = action
        final = self.run_stages(['02_pitch_adaptation'])
        self.assertTrue(final['stages']['02_pitch_adaptation']['attempt'].endswith('attempt_003'))
        self.assertNotIn('error', final)

    def test_run_lock_prevents_two_writers(self):
        with self.pipeline():
            with self.assertRaises(BlockingIOError): self.pipeline()

    def test_legacy_state_is_not_modified(self):
        run = self.root / 'run'
        run.mkdir()
        path = run / 'stage_state.json'
        path.write_text('{"stages": {}, "status":"completed"}')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'legacy'): self.pipeline()
        self.assertEqual(before, path.read_bytes())

    def test_success_marker_tamper_is_not_trusted(self):
        state = self.run_stages()
        marker = Path(state['stages']['01_source_score']['attempt']) / 'outputs.json'
        marker.write_text('{}')
        with self.assertRaises(DependencyError): self.run_stages(['02_pitch_adaptation'])

    def test_no_review_plan_and_dependency_order(self):
        selected = select_stages({'review':False})
        self.assertEqual(len(selected), 9)
        self.assertNotIn('07_stem_separation', selected)
        with self.assertRaises(ValueError): select_stages({'review':False}, stage='09')
        with self.assertRaises(ValueError): select_stages({'review':True}, start='09', until='03')
        ids = active_ids({'review':True})
        for module in load_modules():
            for dependency in module.requires({'review':True, 'transpose':'auto'}):
                self.assertLess(ids.index(dependency), ids.index(module.ID))

    def legacy_alias_cache(self):
        from music_cover.common.io import identity, write_json
        from music_cover.audio import sha256
        state = self.run_stages()
        alias = self.root / 'alias'; alias.symlink_to(self.root, target_is_directory=True)
        record = state['stages']['00_prepare']; attempt = Path(record['attempt'])
        config = json.loads((attempt / 'config.json').read_text())
        code = config['provenance']['code']
        digest = code.pop('00_prepare.py'); code[str(alias / '00_prepare.py')] = digest
        record['fingerprint'] = identity(config['provenance'])
        write_json(attempt / 'config.json', config)
        manifest = json.loads((attempt / 'outputs.json').read_text())
        manifest['fingerprint'] = record['fingerprint']; write_json(attempt / 'outputs.json', manifest)
        record['outputs_manifest_sha256'] = sha256(attempt / 'outputs.json')
        write_json(self.root / 'run/stage_state.json', state)
        return attempt

    def test_legacy_code_path_alias_reuses_only_identical_provenance(self):
        self.legacy_alias_cache()
        self.run_stages(['02_pitch_adaptation'])
        self.assertEqual(len(self.calls), 3)
        Path(self.modules[0].__file__).write_text('# actual code change\n')
        with self.assertRaisesRegex(DependencyError, '00_prepare'):
            self.run_stages(['02_pitch_adaptation'])

    def test_alias_compatibility_rejects_tampered_provenance(self):
        from music_cover.common.io import write_json
        attempt = self.legacy_alias_cache()
        config = json.loads((attempt / 'config.json').read_text())
        config['provenance']['params']['injected'] = True
        write_json(attempt / 'config.json', config)
        with self.assertRaisesRegex(DependencyError, '00_prepare'):
            self.run_stages(['02_pitch_adaptation'])


class DisabledPitchTests(unittest.TestCase):
    def test_zero_transpose_preserves_score_without_loading_models(self):
        from music_cover.modules import pitch_adaptation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.abc'
            source.write_text('X:1\nM:4/4\nL:1/4\nQ:1/4=120\nK:C\nV:Vocal\nC D E F|\n')
            out = root / 'artifacts'
            out.mkdir()
            ctx = SimpleNamespace(execute=lambda *args: self.fail('Disabled transposition must not load a model'))
            paths = pitch_adaptation.run(ctx, out, {'score':source}, {'transpose':'0'})
            self.assertEqual(paths['score'].read_bytes(), source.read_bytes())
            self.assertEqual(json.loads(paths['plan'].read_text())['mode'], 'disabled')
            self.assertNotIn('profiles', paths)


class PublicationTests(unittest.TestCase):
    setUp = ModularExecutionTests.setUp
    tearDown = ModularExecutionTests.tearDown
    pipeline = ModularExecutionTests.pipeline
    run_stages = ModularExecutionTests.run_stages
    def test_failed_replacement_keeps_published_delivery_and_manual_reviews(self):
        delivery = self.modules[-1]
        delivery.ID = '11_delivery'
        def produce(ctx, out, inputs, params):
            path = out / 'result.txt'
            path.write_text(params['variant'])
            review = out / 'review'
            review.mkdir()
            (review / 'review_scores.csv').write_text('reviewer,asset\n,../final.wav\n')
            return {'result': path}
        delivery.run = produce
        ids = [m.ID for m in self.modules]
        self.run_stages(ids)
        published = self.root / 'run/deliverables'
        self.assertTrue(published.is_symlink())
        old_target = published.resolve()
        manual = self.root / 'run/reviews/review_scores.csv'
        manual.write_text('expert notes that must survive')
        delivery.run = lambda ctx,out,i,p: {'missing': out / 'missing.wav'}
        with self.assertRaises(RuntimeError): self.run_stages(ids, request={**self.request, 'variant':'two'})
        self.assertEqual(published.resolve(), old_target)
        self.assertEqual((published / 'result.txt').read_text(), 'one')
        delivery.run = produce
        state = self.run_stages(ids, request={**self.request, 'variant':'two'})
        self.assertEqual(state['status'], 'completed')
        self.assertEqual((published / 'result.txt').read_text(), 'two')
        self.assertTrue(old_target.is_dir())
        self.assertEqual(manual.read_text(), 'expert notes that must survive')


if __name__ == '__main__': unittest.main()
