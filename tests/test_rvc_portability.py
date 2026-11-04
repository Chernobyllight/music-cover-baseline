"""RVC mount-independent training lists, exports and base entry points."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from music_cover.common.rvc_paths import portable_config, runtime_filelist
from music_cover.common.io import write_json
from music_cover.audio import sha256
from music_cover import rvc_training


class FileLists(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root/'dataset'
        self.paths = ['train/0000.wav', 'features/0000.content.npy', 'features/0000.coarse.npy', 'features/0000.f0.npy']
        for name in self.paths:
            f = self.folder/name; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(b'audio/feature')

    def test_dataset_relative_list_survives_move_without_hash_change(self):
        (self.folder/'train.txt').write_text('|'.join([*self.paths, '0'])+'\n')
        before = sha256(self.folder/'train.txt')
        moved = self.root/'new-mount/dataset'; moved.parent.mkdir()
        shutil.move(self.folder, moved)
        old = Path.cwd()
        try:
            os.chdir('/')
            result = runtime_filelist(moved, self.root/'run/runtime_train.txt', 'dataset')
        finally:
            os.chdir(old)
        self.assertEqual(result.read_text(), '|'.join([*(str(moved/x) for x in self.paths), '0'])+'\n')
        self.assertEqual(before, sha256(moved/'train.txt'))

    def test_legacy_list_relocates_from_its_recorded_dataset_root(self):
        original = '|'.join([*('/retired/project/data/dataset/'+x for x in self.paths), '0'])+'\n'
        (self.folder/'train.txt').write_text(original)
        result = runtime_filelist(self.folder, self.root/'runtime.txt')
        self.assertEqual(result.read_text(), '|'.join([*(str(self.folder/x) for x in self.paths), '0'])+'\n')
        self.assertEqual((self.folder/'train.txt').read_text(), original)

    def test_list_cannot_escape_dataset_or_mix_absolute_roots(self):
        for paths, base in [(['../outside.wav', *self.paths[1:]], 'dataset'),
                            ([str(self.folder/x) for x in self.paths], 'dataset'),
                            (['/old/train/0000.wav', '/other/features/0000.content.npy', '/old/'+self.paths[2], '/old/'+self.paths[3]], None)]:
            with self.subTest(paths=paths):
                (self.folder/'train.txt').write_text('|'.join([*paths, '0'])+'\n')
                with self.assertRaises(ValueError): runtime_filelist(self.folder, self.root/'runtime.txt', base)

    def test_invalid_list_and_speaker_rejected(self):
        for value in ['', 'only|two', '|'.join([*self.paths, '1'])]:
            (self.folder/'train.txt').write_text(value)
            with self.assertRaises(ValueError):runtime_filelist(self.folder,self.root/'runtime.txt','dataset')


class Metadata(unittest.TestCase):
    def test_config_normalizes_project_paths_and_keeps_explicit_external_inputs(self):
        root=Path('/new/project')
        original={'source_dataset':'/external/recordings', 'dataset_dir':str(root/'data/dataset'),
                  'export_dir':'models/voice', 'validation':{'cases':[{'source_run':str(root/'runs/song'),'start':1,'duration':2}]}}
        value=portable_config(original,root)
        self.assertEqual(value['source_dataset'],'/external/recordings')
        self.assertEqual(value['dataset_dir'],'data/dataset')
        self.assertEqual(value['validation']['cases'][0]['source_run'],'runs/song')
        self.assertEqual(original['dataset_dir'],str(root/'data/dataset'))

    def test_validation_job_paths_are_portable(self):
        with patch.object(rvc_training,'ROOT',Path('/project')):
            job=rvc_training.portable_job({'source':'/project/runs/case/source.wav','converted':'/project/runs/case/converted.wav',
                    'output':'/project/runs/case/metrics','checkpoint':'/project/runs/train/model.pth',
                    'references':['/project/data/ref.wav'],'metrics':{'speaker_similarity':.5}})
        self.assertEqual(job['checkpoint'],'runs/train/model.pth')
        self.assertEqual(job['references'],['data/ref.wav'])
        self.assertEqual(job['metrics'],{'speaker_similarity':.5})

    def test_export_after_project_move_resolves_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'old'; run=root/'runs/train'; run.mkdir(parents=True)
            files={'model.pth':b'model','model_config.json':b'{}','added.index':b'index','index_manifest.json':b'{}'}
            for name,value in files.items():(run/name).write_bytes(value)
            report={'evaluation_split':'validation','dataset_identity':'d','selected':{
                'eligible':True,'checkpoint':'runs/train/model.pth','checkpoint_sha256':sha256(run/'model.pth'),'index_rate':.5}}
            write_json(run/'validation.json',report)
            write_json(root/'configs/rvc_assets.lock.json',{'assets':[]})
            moved=root.with_name('moved');shutil.move(root,moved);run=moved/'runs/train'
            cfg={'voice_id':'singer','version':'new','export_dir':'models/voice','inference':{'index_rate':.5}}
            old=Path.cwd()
            try:
                os.chdir('/')
                with patch.object(rvc_training,'ROOT',moved),patch.object(rvc_training,'prepared',return_value={'identity':'d'}):
                    rvc_training.export(cfg,run,run/'out',None)
            finally:os.chdir(old)
            profile=json.loads((moved/'models/voice/voice.json').read_text())
            self.assertEqual(profile['training_run'],'runs/train')
            self.assertEqual(profile['checkpoint']['path'],'model.pth')
            self.assertEqual((moved/'models/voice/model.pth').read_bytes(),b'model')

    def test_rvc_entry_points_do_not_delegate_to_another_environment(self):
        root=Path(__file__).resolve().parents[1]
        scripts=[root/'scripts/05_voice_conversion/rvc/run.sh',*(root/'scripts/05_voice_conversion/rvc/training').glob('*/run.sh')]
        self.assertEqual(len(scripts),8)
        for file in scripts:
            text=file.read_text()
            self.assertIn('/opt/conda/bin/python',text)
            self.assertIn('export PATH="/opt/conda/bin:$PATH"',text)
            self.assertNotIn('run_baseline.sh',text)
            self.assertNotIn('/envs/',text)


if __name__=='__main__':unittest.main()
