"""SVC compatibility, strict checkpoint loading and routing contracts (no GPU required)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from music_cover.svc import checked_state, validate_settings, resolve_svc_voice, check_model_config, UPSTREAM_COMMIT
from music_cover.voice import resolve_voice
from music_cover.audio import sha256
from music_cover.common.io import write_json
from music_cover.modules import voice_conversion as stage


class RoutingTests(unittest.TestCase):
    def config(self):return {**stage.DEFAULTS,'backend':'so_vits_svc','mode':'finetuned','voice_profile':'voice.json','candidate_count':1}
    def test_svc_keeps_pitch_and_has_no_diffusion_steps(self):
        p=stage.params({'seed':42,'steps':30},self.config())
        self.assertEqual(p['semitones'],0);self.assertFalse(p['auto_f0']);self.assertIsNone(p['steps'])
        self.assertEqual(stage.backends({},self.config()),['so_vits_svc','voice_metrics'])
    def test_zero_shot_svc_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'finetuned'):stage.params({'seed':42,'steps':30},{**self.config(),'mode':'zero_shot'})
    def test_svc_multiple_candidates_are_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError,'candidates 1'):stage.params({'seed':42,'steps':30},{**self.config(),'candidate_count':3})
    def test_svc_parameters_do_not_leak_into_seed_vc(self):
        with self.assertRaisesRegex(ValueError,'require'):stage.params({'seed':42,'steps':30},{**stage.DEFAULTS,'svc':{'noise_scale':.4}})
    def test_settings_reject_unknown_and_nonfinite(self):
        for p in [{'semitones':-6},{'noise_scale':float('nan')},{'noise_scale':True},{'overlap_seconds':20},{'pad_seconds':0}]:
            with self.subTest(p=p),self.assertRaises(ValueError):validate_settings(p)
    def test_seed_vc_three_candidates_unchanged(self):
        c={**stage.DEFAULTS,'mode':'finetuned','voice_profile':'seed.json'}
        p=stage.params({'seed':42,'steps':30},c)
        self.assertEqual(p['candidate_seeds'],[42,43,44]);self.assertEqual(p['steps'],30)


class LoadingTests(unittest.TestCase):
    def model(self):
        state={'pre.weight':SimpleNamespace(shape=(2,3)),'emb_g.weight':SimpleNamespace(shape=(1,3))}
        return SimpleNamespace(state_dict=lambda:state,load_state_dict=lambda d,strict:None),state
    def test_only_speaker_embedding_may_be_initialized(self):
        model,state=self.model();report=checked_state(model,{'pre.weight':state['pre.weight']},replace_speaker=True)
        self.assertEqual(report['initialized_tensors'],['emb_g.weight'])
    def test_inference_requires_embedding(self):
        model,state=self.model()
        with self.assertRaisesRegex(ValueError,'keys differ'):checked_state(model,{'pre.weight':state['pre.weight']})
    def test_missing_core_tensor_is_never_silently_random(self):
        model,state=self.model()
        with self.assertRaisesRegex(ValueError,'keys differ'):checked_state(model,{},replace_speaker=True)
    def test_incompatible_shape_is_rejected(self):
        model,state=self.model()
        with self.assertRaisesRegex(ValueError,'shape mismatch'):checked_state(model,{**state,'pre.weight':SimpleNamespace(shape=(2,2))})


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.folder=self.root/'voice';self.folder.mkdir()
        config={'data':{'sampling_rate':44100,'hop_length':512},'model':{'speech_encoder':'vec768l12','ssl_dim':768,'gin_channels':768,'n_speakers':1,'vocoder_name':'nsf-hifigan'},'spk':{'yang_kun':0}}
        (self.folder/'model.pth').write_bytes(b'model');write_json(self.folder/'config.json',config);write_json(self.folder/'validation.json',{})
        write_json(self.root/'configs/so_vits_svc_assets.lock.json',{'assets':[]})
        self.profile={'schema_version':1,'backend':'so_vits_svc','adaptation_scope':'speaker','voice_id':'yang_kun','version':'v1',
                      'speaker_name':'yang_kun','upstream_commit':UPSTREAM_COMMIT,'inference_assets':{}}
        for k,n in [('checkpoint','model.pth'),('config','config.json'),('validation','validation.json')]:self.profile[k]={'path':n,'sha256':sha256(self.folder/n)}
        self.save()
    def tearDown(self):self.temp.cleanup()
    def save(self):write_json(self.folder/'voice.json',self.profile)
    def test_dispatch_and_no_inference_reference(self):
        v=resolve_voice('finetuned',self.folder/'voice.json',root=self.root,backend='so_vits_svc');self.assertFalse(v['reference_required'])
    def test_cross_backend_profile_rejected(self):
        with self.assertRaisesRegex(ValueError,'schema/backend'):resolve_voice('finetuned',self.folder/'voice.json',root=self.root)
    def test_changed_weights_rejected(self):
        (self.folder/'model.pth').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):resolve_svc_voice('finetuned',self.folder/'voice.json',self.root)
    def test_changed_encoder_identity_rejected(self):
        self.profile['inference_assets']={'different':'hash'};self.save()
        with self.assertRaisesRegex(ValueError,'assets differ'):resolve_svc_voice('finetuned',self.folder/'voice.json',self.root)
    def test_singer_not_in_model_rejected(self):
        self.profile['speaker_name']='missing';self.save()
        with self.assertRaisesRegex(ValueError,'Speaker missing'):resolve_svc_voice('finetuned',self.folder/'voice.json',self.root)
    def test_matched_vec256_profile_is_accepted(self):
        p=self.folder/'config.json';c=json.loads(p.read_text())
        c['model'].update(speech_encoder='vec256l9',ssl_dim=256,gin_channels=256)
        write_json(p,c);self.profile['config']['sha256']=sha256(p);self.save()
        self.assertEqual(resolve_svc_voice('finetuned',self.folder/'voice.json',self.root)['speaker_name'],'yang_kun')
    def test_mismatched_encoder_dimensions_are_rejected(self):
        original=json.loads((self.folder/'config.json').read_text())
        for patch in [{'speech_encoder':'vec256l9'}, {'ssl_dim':256}, {'speech_encoder':'whisper-ppg'}]:
            config=copy.deepcopy(original);config['model'].update(patch)
            with self.subTest(patch=patch),self.assertRaisesRegex(ValueError,'matched'):check_model_config(config)
    def test_experimental_volume_decoder_cannot_silently_enter_standard_route(self):
        config=json.loads((self.folder/'config.json').read_text());config['model']['vol_embedding']=True
        with self.assertRaisesRegex(ValueError,'non-volume'):check_model_config(config)


class SharedRunTests(unittest.TestCase):
    def test_fork_verifies_upstream_and_never_copies_later_stages(self):
        import importlib.util
        script=Path(__file__).resolve().parents[1]/'scripts/fork_voice_run.py'
        spec=importlib.util.spec_from_file_location('fork_voice_run',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from music_cover.modules import IDS
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir();stages={}
            for name in IDS[:5]:
                attempt=source/name;attempt.mkdir();payload=attempt/'audio.wav';payload.write_bytes(name.encode())
                marker=attempt/'outputs.json';write_json(marker,{'stage':name})
                stages[name]={'status':'succeeded','attempt':str(attempt),'outputs_manifest_sha256':sha256(marker),
                              'outputs':{'audio':{'path':str(payload),'bytes':payload.stat().st_size,'sha256':sha256(payload)}}}
            stages[IDS[5]]={'status':'succeeded','private':'must not copy'}
            write_json(source/'stage_state.json',{'schema_version':2,'stages':stages});write_json(source/'request.json',{});write_json(source/'config.resolved.json',{})
            digest=sha256(source/'stage_state.json');module.fork(source,root/'child')
            child=json.loads((root/'child/stage_state.json').read_text());self.assertEqual(set(child['stages']),set(IDS[:5]))
            self.assertEqual(digest,sha256(source/'stage_state.json'))
            with self.assertRaisesRegex(ValueError,'already exists'):module.fork(source,root/'child')
            next(source.rglob('audio.wav')).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'artifact changed'):module.fork(source,root/'bad')
            self.assertFalse((root/'bad').exists())

if __name__=='__main__':unittest.main()
