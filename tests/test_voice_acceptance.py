"""Quality rejection must preserve generated audio, mixes and review reports."""
import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import soundfile as sf
from music_cover.audio import sha256, inspect_audio, read_at, shift_audio
from music_cover.common.io import write_json
from music_cover.voice_candidates import assessment, select_output, selection_fields
from music_cover.voice_alignment import assess_alignment, onset_alignment
from music_cover.backends import voice_candidates as backend
from music_cover.backends.voice_metrics import VoiceMetrics
from music_cover.modules import mix, delivery
from music_cover.voice_mixing import mix_tracks


def diagnostic(status='unassessable'):
    return {'speaker_similarity':.65,'pitch':{'status':status,'within_one_semitone_fraction':.99},
            'alignment':{'consistent':False,'status':'unassessable'},
            'audio':{'duration_s':6.,'sample_peak':.2,'rms':.1}}


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rate=1000
        rng=np.random.default_rng(9); envelope=np.zeros(8000)
        for t in range(2300,7800,29):envelope[t:t+10]=rng.uniform(.1,.8)
        self.source=(np.repeat(envelope,10)*np.sin(np.arange(80000)*.4))[:,None]
    def test_active_windows_ignore_quiet_ends_and_verify_constant_delay(self):
        result=assess_alignment(self.source,shift_audio(self.source,120,len(self.source))*.7,self.rate)
        self.assertTrue(result['consistent']);self.assertAlmostEqual(result['estimated_delay_ms'],120,delta=10)
        self.assertTrue(result['excluded_low_activity_windows'])
    def test_variable_delay_is_still_flagged_not_relaxed(self):
        other=self.source.copy();other[40000:60000]=shift_audio(self.source,120,len(self.source))[40000:60000]
        result=assess_alignment(self.source,other,self.rate)
        self.assertFalse(result['consistent']);self.assertEqual(result['applied_shift_samples'],0)
    def test_silence_and_unrelated_audio_are_not_verified(self):
        for other in [self.source*0,np.random.default_rng(3).normal(0,.1,self.source.shape)]:
            self.assertFalse(assess_alignment(self.source,other,self.rate)['consistent'])
    def test_onsets_recover_timbre_sensitive_envelope_failure(self):
        primary={'consistent':False,'status':'needs_review'}
        with patch('music_cover.voice_alignment.active_envelope_alignment',return_value=primary):
            result=assess_alignment(self.source,self.source*.7,self.rate)
        self.assertTrue(result['consistent']);self.assertEqual(result['original_envelope_check'],primary)


class CandidateGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        t=np.arange(6*48000)/48000
        voice=(.1*(1+.5*np.sin(2*np.pi*1.7*t))*np.sin(2*np.pi*220*t)).astype(np.float32)
        backing=np.column_stack([.02*np.sin(2*np.pi*110*t),.02*np.sin(2*np.pi*165*t)]).astype(np.float32)
        for name,audio in [('source.wav',voice),('reference.wav',voice),('backing.wav',backing),('cover.wav',backing+voice[:,None])]:
            sf.write(self.root/name,audio,48000,subtype='FLOAT')
        self.request={'backend':'so_vits_svc','voice_profile':'profile.json','seed':42,'candidate_count':1,
                      'candidate_seeds':[42],'source':str(self.root/'source.wav'),'reference':str(self.root/'reference.wav'),
                      'steps':None,'timeout_seconds':10}
    def generate(self,*,metric=None,error=None,backend_name='so_vits_svc',fail_inference=False):
        out=self.root/backend_name;out.mkdir()
        self.request['backend']=backend_name
        def infer(command,**kwargs):
            if fail_inference:raise subprocess.CalledProcessError(1,command)
            folder=Path(command[command.index('--output')+1]);shutil.copy2(self.root/'source.wav',folder/'converted.wav')
            write_json(folder/'conversion.json',{'backend':backend_name})
        evaluator=SimpleNamespace(compare=lambda *a:copy.deepcopy(metric or diagnostic()))
        with patch.object(backend,'resolve_voice',return_value={'profile':Path('profile.json')}),patch.object(backend,'voice_record',return_value={}),patch.object(backend.subprocess,'run',side_effect=infer),patch('music_cover.backends.voice_metrics.VoiceMetrics',side_effect=error,return_value=evaluator):
            backend.generate(self.request,out)
        return out,json.loads((out/'candidates.json').read_text())
    def test_all_three_backends_deliver_rejected_audio_without_recommending(self):
        for name in ['seed_vc','rvc','so_vits_svc']:
            with self.subTest(backend=name):
                out,report=self.generate(backend_name=name)
                self.assertEqual(report['generation_status'],'succeeded');self.assertEqual(report['status'],'review_required')
                self.assertIsNone(report['recommended_id']);self.assertEqual(report['selected_id'],'candidate_001')
                self.assertFalse(report['candidates'][0]['eligible']);self.assertEqual(sha256(out/'converted.wav'),sha256(self.root/'source.wav'))
                self.assertIn('未宣称质量合格',(out/'candidates.html').read_text())
    def test_metric_initialization_failure_retains_audio_and_reason(self):
        out,report=self.generate(error=RuntimeError('missing evaluation model'))
        self.assertTrue((out/'converted.wav').exists());self.assertEqual(report['candidates'][0]['evaluation_status'],'failed')
        self.assertIn('missing evaluation model',report['candidates'][0]['evaluation_error'])
        self.assertIsNone(report['recommended_id'])
    def test_nonfinite_metric_is_reported_missing_without_losing_audio(self):
        metric=diagnostic();metric['speaker_similarity']=float('nan')
        out,report=self.generate(metric=metric)
        self.assertTrue((out/'converted.wav').exists())
        self.assertIsNone(report['candidates'][0]['metrics']['speaker_similarity'])
        self.assertIn('invalid_ecapa',report['candidates'][0]['rejection_reasons'])
    def test_final_voice_evaluation_error_is_a_report_not_audio_failure(self):
        from music_cover.backends.voice_metrics import evaluate_voice
        with patch('music_cover.backends.voice_metrics.VoiceMetrics',side_effect=RuntimeError('evaluation unavailable')):
            evaluate_voice({'source':'source','converted':'converted','reference':'reference'},self.root/'eval')
        report=json.loads((self.root/'eval/voice_conversion_evaluation.json').read_text())
        self.assertEqual(report['assessment']['status'],'error');self.assertFalse(report['assessment']['blocks_delivery'])

    def test_inference_failure_is_not_disguised_as_quality_warning(self):
        with self.assertRaisesRegex(RuntimeError,'No readable generated audio'):self.generate(fail_inference=True)
        self.assertFalse((self.root/'so_vits_svc/converted.wav').exists())
    def test_rejected_audio_produces_real_mix_and_delivery_files(self):
        out,report=self.generate();mixed=self.root/'mix';mixed.mkdir()
        inputs={'cover':self.root/'cover.wav','vocals':self.root/'source.wav','instrumental':self.root/'backing.wav',
                'converted':out/'converted.wav','candidate_report':out/'candidates.json'}
        outputs=mix.run(None,mixed,inputs,mix.DEFAULTS)
        self.assertTrue((mixed/'final.mp3').stat().st_size>0)
        result=json.loads((mixed/'candidates.json').read_text());self.assertIsNone(result['recommended_id'])
        self.assertEqual(result['candidates'][0]['mix_status'],'succeeded')
        self.assertTrue(json.loads((mixed/'mix.json').read_text())['delivery_requires_review'])
        target=self.root/'delivered';target.mkdir();evaluation=self.root/'evaluation.json';write_json(evaluation,{'voice_candidates':result})
        delivery_inputs={'06_mix/'+k:v for k,v in outputs.items()};delivery_inputs['10_evaluation/evaluation.json']=evaluation
        def build(out,request,stages,prepared,generated,separated,mixed,*a,**kw):
            for key in ['final','final_mp3']:shutil.copy2(mixed[key],out/mixed[key].name)
            (out/'listen.html').write_text('<h1>listen</h1>');(out/'report.md').write_text('report\n')
        from music_cover.common.paths import RunPaths
        ctx=SimpleNamespace(request={'transpose':'0','review':False},state={'stages':{}},config={},execute=lambda *a:None,paths=RunPaths(self.root,self.root/'run'))
        with patch.object(delivery,'build_delivery',build):delivery.run(ctx,target,delivery_inputs,{})
        self.assertEqual(sha256(target/'final.mp3'),sha256(target/'candidates/candidate_001/final.mp3'))
        self.assertIn('无合格候选',(target/'report.md').read_text())
    def test_low_quality_duration_and_silence_still_mix(self):
        for name,audio in [('long',np.pad(read_at(self.root/'source.wav'),((0,48000),(0,0)))),('silent',np.zeros((6*48000,1)))]:
            file=self.root/(name+'.wav');sf.write(file,audio,48000,subtype='FLOAT');out=self.root/name;out.mkdir()
            result=mix_tracks(self.root/'cover.wav',self.root/'source.wav',self.root/'backing.wav',file,out)
            self.assertTrue((out/'final.mp3').exists());self.assertTrue(result['delivery_requires_review'])
            self.assertEqual(result['final']['frames'],len(audio))
    def test_same_generated_audio_unchanged_after_evaluation(self):
        original=sha256(self.root/'source.wav');out,report=self.generate()
        self.assertEqual(original,sha256(self.root/'source.wav'));self.assertEqual(original,report['candidates'][0]['audio_sha256'])
    def test_uncertain_alignment_is_not_reported_as_wrong_pitch(self):
        evaluator=VoiceMetrics.__new__(VoiceMetrics)
        evaluator.pitch=lambda *a:None;evaluator.embedding=lambda *a:np.array([1.,0.])
        uncertain={'consistent':False,'status':'unassessable','applied_shift_samples':0}
        target=self.root/'diagnostics'
        with patch('music_cover.backends.voice_metrics.assess_alignment',return_value=uncertain),patch('music_cover.pitch_validation.compare_conversion',return_value={'status':'verified','observed_semitones':0.}):
            result=evaluator.compare(self.root/'source.wav',self.root/'source.wav',[self.root/'reference.wav'],target)
        self.assertEqual(result['pitch']['unaligned_estimate_status'],'verified')
        self.assertEqual(result['pitch']['status'],'unassessable')
        self.assertEqual(assessment(result,6.)['status'],'unassessable')
    def test_verified_candidate_has_priority_over_rejected_higher_ecapa(self):
        rows=[{'id':'bad','seed':42,'status':'succeeded','generation_status':'succeeded','audio':'bad.wav','eligible':False,'metrics':{'speaker_similarity':.99}},
              {'id':'good','seed':43,'status':'succeeded','audio':'good.wav','eligible':True,'metrics':{'speaker_similarity':.2}}]
        self.assertEqual(select_output(rows)['id'],'good')
        rows[1]['eligible']=False;self.assertEqual(select_output(rows)['id'],'bad')
        rows[0]['metrics']={};rows[1]['metrics']={};self.assertEqual(select_output(rows)['id'],'bad')


class ExportReviewTests(unittest.TestCase):
    def test_svc_rejected_model_can_export_with_explicit_review_status(self):
        import torch
        from music_cover import svc_training as module
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';run.mkdir();out=root/'out'
            checkpoint=run/'checkpoint.pth';torch.save({'model':{'weight':torch.zeros(1)},'step':4000},checkpoint)
            write_json(run/'model_config.json',{})
            write_json(run/'validation/validation.json',{'dataset_identity':'d','evaluation_split':'validation','selected_checkpoint':str(checkpoint),
                    'candidates':[{'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'eligible':False}]})
            write_json(run/'state.json',{'stages':{'02_validate':{'output':str(run/'validation')}}})
            c={'voice_id':'singer','version':'v1','export_dir':'voice','training':{'learning_rate':1e-4}}
            with patch.object(module,'ROOT',root),patch.object(module,'prepared',return_value={'identity':'d'}),patch.object(module,'check_assets',return_value={'assets':[]}):
                module.export(c,run,out,None)
            profile=json.loads((root/'voice/voice.json').read_text());self.assertEqual(profile['assessment_status'],'review_required')
            self.assertEqual(profile['selected_step'],4000)
    def test_rvc_rejected_model_can_export_without_changing_checkpoint(self):
        from music_cover import rvc_training as module
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';run.mkdir();out=root/'out'
            for name in ['model.pth','model_config.json','added.index','index_manifest.json']:(run/name).write_bytes(b'model')
            checkpoint=run/'model.pth';write_json(run/'validation.json',{'evaluation_split':'validation','dataset_identity':'d','selected':{
                    'eligible':False,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'index_rate':.5}})
            write_json(root/'configs/rvc_assets.lock.json',{'assets':[]})
            c={'voice_id':'singer','version':'v1','export_dir':'voice','inference':{'index_rate':.5}}
            with patch.object(module,'ROOT',root),patch.object(module,'prepared',return_value={'identity':'d'}):module.export(c,run,out,None)
            profile=json.loads((root/'voice/voice.json').read_text());self.assertEqual(profile['assessment_status'],'review_required')
            self.assertEqual(sha256(root/'voice/model.pth'),sha256(checkpoint))

if __name__=='__main__':unittest.main()
