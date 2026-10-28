"""Candidate ranking, mode isolation and actual WAV/MP3 mixing contracts."""
import copy
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
from music_cover.voice_candidates import candidate_seeds, rejection_reasons, recommend, contained_file, render_report
from music_cover.modules import voice_conversion, mix, delivery


def metrics(score=.5):
    return {'speaker_similarity': score, 'pitch': {'status': 'verified', 'within_one_semitone_fraction': .98},
            'alignment': {'consistent': True}, 'audio': {'duration_s': 6., 'sample_peak': .2, 'rms': .1}}


class SelectionTests(unittest.TestCase):
    def test_seeds_deterministic_and_wrap(self):
        self.assertEqual(candidate_seeds(42), [42,43,44])
        self.assertEqual(candidate_seeds(2**32-1), [2**32-1,0,1])
        self.assertEqual(candidate_seeds(42, 3, [10,20,30]), [10,20,30])
    def test_bad_settings_fail(self):
        for count,seeds in [(0,None),(9,None),(True,None),(3,[42,42,44]),(3,[42,43]),(3,[42,43,-1]),(3,[42,43,2**32])]:
            with self.subTest(count=count,seeds=seeds), self.assertRaises(ValueError): candidate_seeds(42,count,seeds)
    def test_zero_shot_does_not_require_candidate_evaluation(self):
        c={**voice_conversion.DEFAULTS,'mode':'zero_shot','voice_profile':'missing','candidate_count':9,'candidate_seeds':[1]}
        p=voice_conversion.params({'steps':30,'seed':42}, c)
        self.assertEqual(p['candidate_count'],1); self.assertEqual(p['candidate_seeds'],[42]); self.assertIsNone(p['voice_profile'])
        self.assertEqual(voice_conversion.backends({},c),['seed_vc'])
    def test_finetuned_requires_evaluation_and_three_candidates(self):
        c={**voice_conversion.DEFAULTS,'mode':'finetuned','voice_profile':'models/voice.json'}
        p=voice_conversion.params({'steps':30,'seed':42},c)
        self.assertEqual(p['candidate_seeds'],[42,43,44]); self.assertIn('voice_metrics',voice_conversion.backends({},c))
    def test_technical_gate(self):
        self.assertEqual(rejection_reasons(metrics(),6.),[])
        for field,value,reason in [('pitch',{'status':'unassessable'},'pitch_not_verified'),('alignment',{'consistent':False},'alignment_not_verified'),('speaker_similarity',float('nan'),'invalid_ecapa')]:
            m=metrics();m[field]=value;self.assertIn(reason,rejection_reasons(m,6.))
        for field,value,reason in [('duration_s',7,'duration_mismatch'),('sample_peak',1.2,'invalid_peak'),('rms',0,'silent_audio')]:
            m=metrics();m['audio'][field]=value;self.assertIn(reason,rejection_reasons(m,6.))
    def test_ranking_ignores_failed_high_score_and_breaks_ties(self):
        rows=[{'id':'a','seed':44,'status':'succeeded','eligible':True,'metrics':metrics(.6)},
              {'id':'b','seed':43,'status':'succeeded','eligible':True,'metrics':metrics(.6)},
              {'id':'c','seed':42,'status':'succeeded','eligible':False,'metrics':metrics(.99)}]
        self.assertEqual(recommend(rows)['id'],'b')
        rows[1]['status']='failed';self.assertEqual(recommend(rows)['id'],'a')
        rows[0]['eligible']=False;self.assertIsNone(recommend(rows))
    def test_profile_cannot_enable_removed_song_mode(self):
        from test_voice_adaptation import ProfileTests
        case=ProfileTests();case.setUp()
        try:
            p=case.profile();data=json.loads(p.read_text());data['schema_version']=2;write_json(p,data)
            from music_cover.voice import resolve_voice
            with self.assertRaisesRegex(ValueError,'schema'): resolve_voice('finetuned',p,root=case.root)
            data['schema_version']=1;data['adaptation_scope']='song_instance';write_json(p,data)
            with self.assertRaisesRegex(ValueError,'speaker-level'): resolve_voice('finetuned',p,root=case.root)
        finally: case.tearDown()
    def test_candidate_change_changes_only_conversion_parameters(self):
        from music_cover.modules import prepare, source_score, pitch_adaptation, style_generation, vocal_separation
        from music_cover.pipeline import DEFAULT_REQUEST
        request={**DEFAULT_REQUEST,'song':'song','lyrics':'lyrics','reference':'ref','style':'style'}
        c={**voice_conversion.DEFAULTS,'mode':'finetuned','voice_profile':'voice.json'}
        first=voice_conversion.params(request,c)
        self.assertNotEqual(first,voice_conversion.params(request,{**c,'candidate_seeds':[11,12,13]}))
        # No candidate settings are introduced into the shared request consumed upstream.
        for module in [prepare, source_score, pitch_adaptation, style_generation, vocal_separation]:
            self.assertNotIn('candidate_seeds',module.params(request,module.DEFAULTS))


class MixDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.raw=self.root/'raw';self.raw.mkdir();self.out=self.root/'mix';self.out.mkdir()
        t=np.arange(6*48000)/48000
        voice=(.1*(1+.5*np.sin(2*np.pi*1.7*t))*np.sin(2*np.pi*220*t)).astype(np.float32)
        backing=np.column_stack([.02*np.sin(2*np.pi*110*t),.02*np.sin(2*np.pi*165*t)]).astype(np.float32)
        sf.write(self.raw/'vocals.wav',voice,48000,subtype='FLOAT')
        sf.write(self.raw/'backing.wav',backing,48000,subtype='FLOAT')
        sf.write(self.raw/'cover.wav',backing+voice[:,None],48000,subtype='FLOAT')
        rows=[]
        for n,score in enumerate([.6,.8,.7],1):
            name=f'candidate_{n:03d}';p=self.raw/'candidates'/name/'converted.wav';p.parent.mkdir(parents=True)
            sf.write(p,voice*(.9+.1*n),48000,subtype='FLOAT')
            rows.append({'id':name,'seed':41+n,'status':'succeeded','eligible':True,'metrics':metrics(score),
                         'audio':str(p.relative_to(self.raw)),'audio_sha256':sha256(p),'rejection_reasons':[]})
        self.report={'candidates':rows,'recommended_id':'candidate_002'};write_json(self.raw/'candidates.json',self.report)
        self.i={'cover':self.raw/'cover.wav','vocals':self.raw/'vocals.wav','instrumental':self.raw/'backing.wav',
                'converted':self.raw/rows[1]['audio'],'candidate_report':self.raw/'candidates.json'}
    def tearDown(self):self.temp.cleanup()
    def test_three_real_mp3_and_delivery_recommendation(self):
        outputs=mix.run(None,self.out,self.i,mix.DEFAULTS)
        report=json.loads(outputs['candidate_report'].read_text());self.assertEqual(report['recommended_id'],'candidate_002')
        for row in report['candidates']:
            for kind in ['final_mp3','final_wav']:
                p=contained_file(self.out,row[kind]);self.assertEqual(sha256(p),row[kind+'_sha256'])
        selected=report['candidates'][1]
        self.assertEqual(sha256(outputs['final_mp3']),sha256(self.out/selected['final_mp3']))
        target=self.root/'delivery';target.mkdir()
        eval_file=self.root/'evaluation.json';write_json(eval_file,{'voice_candidates':report})
        inputs={'06_mix/'+k:v for k,v in outputs.items()};inputs['10_evaluation/evaluation.json']=eval_file
        def build(out,request,stages,prepared,generated,separated,mixed,*args,**kwargs):
            import shutil
            for key in ['final','final_mp3']:shutil.copy2(mixed[key],out/mixed[key].name)
            (out/'listen.html').write_text('<h1>listen</h1>');(out/'report.md').write_text('report\n')
        ctx=SimpleNamespace(request={'transpose':'0','review':False},state={'stages':{}},config={},execute=lambda *a:None)
        with patch.object(delivery,'build_delivery',build):delivery.run(ctx,target,inputs,{})
        self.assertEqual(sha256(target/'final.mp3'),sha256(target/selected['final_mp3']))
        self.assertIn('candidates.html',(target/'listen.html').read_text())
        self.assertEqual(len(list((target/'candidates').rglob('final.mp3'))),3)
    def test_mismatched_recommendation_is_rejected(self):
        self.report['recommended_id']='candidate_001';write_json(self.raw/'candidates.json',self.report)
        with self.assertRaisesRegex(ValueError,'recommendation'):mix.run(None,self.out,self.i,mix.DEFAULTS)
    def test_paths_cannot_escape_report_directory(self):
        with self.assertRaises(ValueError):contained_file(self.out,'../raw/vocals.wav')
        with self.assertRaises(ValueError):contained_file(self.out,'missing.wav')
    def test_no_eligible_report_can_be_rendered(self):
        self.report['recommended_id']=None
        for row in self.report['candidates']:row['eligible']=False
        render_report(self.report,self.out/'candidates.html')
        self.assertIn('无合格候选',(self.out/'candidates.html').read_text())

if __name__ == '__main__':unittest.main()
