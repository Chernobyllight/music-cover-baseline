"""Routing, weights, isolation and portable artifacts for the Fish backend."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import soundfile as sf
from music_cover.audio import sha256
from music_cover.common.io import write_json
from music_cover.common.voice_paths import training_output
from music_cover.fish import (MODEL_SPEC,UPSTREAM_COMMIT,validate_settings,checked_state,resolve_fish_voice)
from music_cover.fish_config import checked_dataset, training_advisories
from music_cover.modules import voice_conversion as stage
from music_cover.voice import merge_voice_config
from music_cover.common.resources import python_for


class FishRouting(unittest.TestCase):
    def config(self):return {**stage.DEFAULTS,'backend':'fish_diffusion','mode':'finetuned','voice_profile':'voice.json','candidate_count':1}
    def test_fish_keeps_pitch_and_does_not_reuse_seed_steps(self):
        p=stage.params({'seed':42,'steps':30},self.config())
        self.assertEqual(p['semitones'],0);self.assertFalse(p['auto_f0']);self.assertIsNone(p['steps'])
        self.assertEqual(p['fish']['sampler_interval'],10)
    def test_reject_zero_shot(self):
        with self.assertRaisesRegex(ValueError,'finetuned'):stage.params({'seed':42,'steps':30},{**self.config(),'mode':'zero_shot'})
    def test_switch_clears_private_presets(self):
        current={**self.config(),'fish':{'sampler_interval':20}}
        p=merge_voice_config(current,{'backend':'seed_vc'})
        self.assertEqual(p['fish'],{});self.assertEqual(p['candidate_count'],3)
        self.assertIsNone(p['voice_profile'])
    def test_foreign_settings_rejected(self):
        with self.assertRaisesRegex(ValueError,'fish settings'):
            stage.params({'seed':42,'steps':30},{**stage.DEFAULTS,'fish':{'sampler_interval':10}})
    def test_existing_seed_candidates_unchanged(self):
        p=stage.params({'seed':42,'steps':30},{**stage.DEFAULTS,'mode':'finetuned','voice_profile':'seed.json'})
        self.assertEqual(p['candidate_seeds'],[42,43,44]);self.assertEqual(p['steps'],30)
    def test_all_voice_interpreters_are_base(self):
        for backend in ('fish_diffusion','seed_vc','so_vits_svc','rvc'):
            self.assertEqual(python_for('/different/mount',backend),Path('/opt/conda/bin/python'))
    def test_bad_sampler_and_chunks(self):
        for value in ({'sampler_interval':True},{'sampler_interval':0},{'sampler_interval':101},
                      {'chunk_seconds':float('nan')},{'overlap_seconds':0},{'pad_seconds':10},
                      {'noise_predictor':'unknown'},{'transpose':6}):
            with self.subTest(value=value),self.assertRaises(ValueError):validate_settings(value)


class FishSchedule(unittest.TestCase):
    def test_warmup_only_is_visible_but_not_an_error(self):
        self.assertIn('smoke-test',training_advisories({'max_steps':1000,'warmup_steps':1000,'decay_steps':5000})[0])
    def test_truncated_decay_is_explained(self):
        self.assertIn('decay endpoint',training_advisories({'max_steps':3000,'warmup_steps':1000,'decay_steps':5000})[0])
    def test_complete_schedule_has_no_advisory(self):
        self.assertEqual(training_advisories({'max_steps':5000,'warmup_steps':1000,'decay_steps':5000}),[])


class FishWeights(unittest.TestCase):
    def model(self):
        state={'weight':SimpleNamespace(shape=(2,3)),'speaker_encoder.embedding.weight':SimpleNamespace(shape=(1,256))}
        return SimpleNamespace(state_dict=lambda:state,load_state_dict=lambda d,strict:None),state
    def test_only_speaker_can_be_reinitialized(self):
        model,state=self.model()
        report=checked_state(model,{'weight':state['weight']},replace_speaker=True)
        self.assertEqual(report['initialized_tensors'],['speaker_encoder.embedding.weight'])
    def test_missing_acoustic_weights_fail(self):
        model,state=self.model()
        with self.assertRaisesRegex(ValueError,'state mismatch'):checked_state(model,{},replace_speaker=True)
    def test_wrong_shape_fails(self):
        model,state=self.model()
        with self.assertRaisesRegex(ValueError,'state mismatch'):checked_state(model,{**state,'weight':SimpleNamespace(shape=(3,3))})
    def test_inference_requires_speaker(self):
        model,state=self.model()
        with self.assertRaises(ValueError):checked_state(model,{'weight':state['weight']})


class FishData(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.rows=[]
        for i,split in enumerate(('train','validation','test')):
            p=self.root/(split+'.wav');sf.write(p,np.sin(np.arange(44100)*(i+1)/100),44100)
            self.rows.append({'path':p.name,'sha256':sha256(p),'group':split,'split':split})
        self.save()
    def tearDown(self):self.temp.cleanup()
    def save(self):write_json(self.root/'manifest.json',{'voice_id':'singer','clips':self.rows})
    def test_clean_split_passes(self):self.assertEqual(len(checked_dataset(self.root,'singer')['clips']),3)
    def test_recording_leakage_fails(self):
        self.rows[1]['group']='train';self.save()
        with self.assertRaisesRegex(ValueError,'leakage'):checked_dataset(self.root,'singer')
    def test_modified_audio_fails(self):
        (self.root/'train.wav').write_bytes(b'changed')
        with self.assertRaises(ValueError):checked_dataset(self.root,'singer')
    def test_wrong_singer_fails(self):
        with self.assertRaises(ValueError):checked_dataset(self.root,'other')


class FishProfile(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.folder=self.root/'model';self.folder.mkdir()
        write_json(self.folder/'config.json',{'model':MODEL_SPEC,'speaker_name':'singer'})
        (self.folder/'model.ckpt').write_bytes(b'model');write_json(self.folder/'validation.json',{'eligible':False})
        write_json(self.root/'configs/fish_diffusion_assets.lock.json',{'assets':[]})
        data={'schema_version':1,'backend':'fish_diffusion','adaptation_scope':'speaker','upstream_commit':UPSTREAM_COMMIT,
              'voice_id':'singer','speaker_name':'singer','version':'test','inference_assets':{},'validation_status':'quality_review_required'}
        for key,name in [('config','config.json'),('checkpoint','model.ckpt'),('validation','validation.json')]:
            data[key]={'path':name,'sha256':sha256(self.folder/name)}
        write_json(self.folder/'voice.json',data)
    def tearDown(self):self.temp.cleanup()
    def test_review_required_model_can_still_infer(self):
        result=resolve_fish_voice('finetuned','model/voice.json',self.root)
        self.assertEqual(result['validation_status'],'quality_review_required')
        self.assertFalse(result['reference_required'])
    def test_relocated_profile_resolves_relative_assets(self):
        dest=self.root/'other_mount';dest.mkdir()
        import shutil
        shutil.copytree(self.folder,dest/'model');shutil.copytree(self.root/'configs',dest/'configs')
        self.assertEqual(resolve_fish_voice('finetuned','model/voice.json',dest)['checkpoint'],dest/'model/model.ckpt')
    def test_corrupt_export_is_rejected(self):
        (self.folder/'model.ckpt').write_bytes(b'corrupted')
        with self.assertRaises(ValueError):resolve_fish_voice('finetuned','model/voice.json',self.root)
    def test_training_state_is_run_relative(self):
        run=self.root/'run';write_json(run/'state.json',{'stages':{'01_features':{'output':'stages/01_features/attempt_001/artifacts'}}})
        self.assertEqual(training_output(run,'01_features'),run/'stages/01_features/attempt_001/artifacts')


if __name__=='__main__':unittest.main()
