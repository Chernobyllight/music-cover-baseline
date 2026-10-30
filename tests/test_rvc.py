"""RVC routing, retrieval numerical stability, profile integrity and split leakage."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from music_cover.rvc import validate_settings,resolve_rvc_voice,UPSTREAM_COMMIT
from music_cover.voice import resolve_voice,voice_record
from music_cover.modules import voice_conversion as stage
from music_cover.backends.rvc import retrieve,pitch_to_coarse
from music_cover.audio import sha256
from music_cover.common.io import write_json


class Routing(unittest.TestCase):
    def config(self):return {**stage.DEFAULTS,'backend':'rvc','mode':'finetuned','voice_profile':'rvc.json'}
    def test_rvc_preserves_pitch_and_supports_three_explicit_seeds(self):
        p=stage.params({'seed':42,'steps':30},self.config())
        self.assertEqual(p['candidate_seeds'],[42,43,44]);self.assertEqual(p['semitones'],0)
        self.assertFalse(p['auto_f0']);self.assertIsNone(p['steps'])
        self.assertEqual(stage.backends({},self.config()),['rvc','voice_metrics'])
    def test_zero_shot_rvc_rejected(self):
        with self.assertRaisesRegex(ValueError,'finetuned'):stage.params({'seed':42,'steps':30},{**self.config(),'mode':'zero_shot'})
    def test_backend_settings_do_not_leak(self):
        for c in [{**stage.DEFAULTS,'rvc':{'index_rate':.5}},{**self.config(),'svc':{'noise_scale':.1}}]:
            with self.assertRaisesRegex(ValueError,'require'):stage.params({'seed':42,'steps':30},c)
    def test_bad_settings(self):
        for s in [{'index_rate':-1},{'index_rate':float('nan')},{'protect':True},{'protect':.6},{'semitones':-6},{'chunk_seconds':1},{'overlap_seconds':8}]:
            with self.subTest(s=s),self.assertRaises(ValueError):validate_settings(s)


class Retrieval(unittest.TestCase):
    def test_zero_distance_neighbors_stay_finite(self):
        import faiss
        x=np.ones((8,768),np.float32);index=faiss.IndexFlatL2(768);index.add(x)
        result=retrieve(x[:1],index,x,.75)
        self.assertTrue(np.isfinite(result).all());np.testing.assert_allclose(result,x[:1])
    def test_missing_index_does_not_silently_disable_retrieval(self):
        with self.assertRaisesRegex(ValueError,'Invalid'):retrieve(np.zeros((1,768),np.float32),None,None,.5)
    def test_index_zero_is_a_real_no_retrieval_control(self):
        x=np.ones((2,768),np.float32)
        np.testing.assert_array_equal(retrieve(x,None,None,0),x)
    def test_wrong_index_dimension_rejected(self):
        from types import SimpleNamespace
        with self.assertRaisesRegex(ValueError,'Invalid'):retrieve(np.ones((1,768),np.float32),SimpleNamespace(d=256),np.ones((8,256)),.5)
    def test_f0_quantization_keeps_original_unvoiced_mask(self):
        x=np.array([0,50,110,440,1100,2000],np.float32);old=x.copy();coarse=pitch_to_coarse(x)
        np.testing.assert_array_equal(x,old);self.assertEqual(coarse[0],1)
        self.assertEqual(coarse[-1],255);self.assertTrue(np.all(np.diff(coarse)>=0))


class Profiles(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.folder=self.root/'voice';self.folder.mkdir()
        write_json(self.root/'configs/rvc_assets.lock.json',{'assets':[]})
        config={'version':'v2','f0':1,'f0_policy':'unvoiced_zero','speaker_id':0,'data':{'sampling_rate':48000,'hop_length':480},'model':{'spk_embed_dim':1}}
        (self.folder/'model.pth').write_bytes(b'weights');(self.folder/'added.index').write_bytes(b'index')
        write_json(self.folder/'config.json',config);write_json(self.folder/'validation.json',{})
        write_json(self.folder/'index_manifest.json',{'dataset_identity':'dataset','split':'train','dimension':768,'index_sha256':sha256(self.folder/'added.index')})
        self.profile={'schema_version':1,'backend':'rvc','adaptation_scope':'speaker','upstream_commit':UPSTREAM_COMMIT,
                      'voice_id':'huang_qi_shan','version':'test','dataset_identity':'dataset','inference_assets':{}}
        for k,n in [('checkpoint','model.pth'),('config','config.json'),('index','added.index'),('index_manifest','index_manifest.json'),('validation','validation.json')]:
            self.profile[k]={'path':n,'sha256':sha256(self.folder/n)}
        self.save()
    def tearDown(self):self.temp.cleanup()
    def save(self):write_json(self.folder/'voice.json',self.profile)
    def test_dispatch_and_index_record(self):
        v=resolve_voice('finetuned',self.folder/'voice.json',self.root,backend='rvc')
        self.assertFalse(v['reference_required']);self.assertIn('index',voice_record(v))
    def test_modified_index_rejected(self):
        (self.folder/'added.index').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):resolve_rvc_voice('finetuned',self.folder/'voice.json',self.root)
    def test_evaluation_index_rejected_even_with_updated_checksum(self):
        p=self.folder/'index_manifest.json';m=json.loads(p.read_text());m['split']='test';write_json(p,m)
        self.profile['index_manifest']['sha256']=sha256(p);self.save()
        with self.assertRaisesRegex(ValueError,'split'):resolve_rvc_voice('finetuned',self.folder/'voice.json',self.root)
    def test_wrong_backend_rejected(self):
        with self.assertRaisesRegex(ValueError,'schema/backend'):resolve_voice('finetuned',self.folder/'voice.json',self.root,backend='seed_vc')
    def test_external_asset_path_rejected(self):
        p=self.root/'outside';p.write_bytes(b'x');self.profile['index']={'path':'../outside','sha256':sha256(p)};self.save()
        with self.assertRaisesRegex(ValueError,'checksum'):resolve_rvc_voice('finetuned',self.folder/'voice.json',self.root)


class DataSplits(unittest.TestCase):
    def test_duplicate_song_group_across_splits_rejected(self):
        import music_cover.rvc_training as training
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'dataset';folder.mkdir();rows=[]
            for split in ['train','test']:
                f=folder/(split+'.wav');f.write_bytes(split.encode());rows.append({'path':f.name,'sha256':sha256(f),'group':'same_song','split':split})
            write_json(folder/'manifest.json',{'voice_id':'huang_qi_shan','clips':rows})
            with patch.object(training,'ROOT',root),self.assertRaisesRegex(ValueError,'leakage'):
                training.source_manifest({'source_dataset':'dataset','voice_id':'huang_qi_shan'})


class FinalModelPolicy(unittest.TestCase):
    def test_default_final_model_can_be_overridden(self):
        from music_cover.rvc_training import validation_checkpoint
        run=Path('/example/run');config={'training':{'max_steps':4000}}
        self.assertEqual(validation_checkpoint(config,run),run/'checkpoints/model_00004000.pth')
        chosen='/example/run/checkpoints/model_00002000.pth'
        self.assertEqual(validation_checkpoint(config,run,chosen),Path(chosen))

    def test_rvc_import_namespace_is_restored_after_error(self):
        import sys,types
        from music_cover.backends.rvc import runtime
        sentinel=types.ModuleType('tools')
        with patch.dict(sys.modules,{'tools':sentinel}):
            with self.assertRaisesRegex(RuntimeError,'simulated'):
                with runtime():
                    self.assertIsNot(sys.modules['tools'],sentinel)
                    raise RuntimeError('simulated')
            self.assertIs(sys.modules['tools'],sentinel)


if __name__=='__main__':unittest.main()
