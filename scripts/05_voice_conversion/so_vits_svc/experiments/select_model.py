"""Freeze validation-only selection before scoring held-out test cases."""
import json,collections,statistics,datetime
from pathlib import Path
from music_cover.common.paths import ROOT
r=ROOT/'runs/svc_optimization_yang_kun_20261028'
if (r/'selection.json').exists():raise ValueError('Selection is already frozen')
groups=collections.defaultdict(list)
for folder in ['final_validation_a','final_validation_b','final_validation_diffusion']:
 d=json.loads((r/folder/'evaluation/results.json').read_text());outputs=json.loads((r/folder/'outputs.json').read_text())
 assert len(d)==len(outputs) and all(x['status']=='succeeded' for x in d)
 for x in d:groups[(x['model'],x['variant'],folder)].append(x)
summary=[]
for (name,variant,folder),rows in groups.items():
 assert len(rows)==10
 c=[x for x in rows if x['role']=='conversion'];s=[x for x in rows if x['role']=='reconstruction'];assert len(c)==4 and len(s)==6
 summary.append({'model':name,'variant':variant,'folder':folder,'ecapa':statistics.mean(x['metrics']['speaker_similarity'] for x in c),'f0_mean':statistics.mean(x['metrics']['pitch']['within_one_semitone_fraction'] for x in c),'f0_min':min(x['metrics']['pitch']['within_one_semitone_fraction'] for x in c),'self_logspec':statistics.mean(x['paired_reconstruction']['multi_resolution_log_spectral_l1'] for x in s),'valid_audio':all(1e-5<x['metrics']['audio']['rms'] and x['metrics']['audio']['sample_peak']<=1.001 for x in rows)})
baseline=next(x for x in summary if x['model']=='svc_1000')
for x in summary:x['eligible']=x['valid_audio'] and x['f0_min']>=.9 and x['self_logspec']<=baseline['self_logspec']*1.05
ranked=sorted([x for x in summary if x['eligible'] and x['model']!='svc_1000'],key=lambda x:x['ecapa'],reverse=True);best=ranked[0]
req=json.loads((r/(best['folder']+'_request.json')).read_text());model=next(m for m in req['models'] if m['id']==best['model']);variant=next(v for v in model['variants'] if v['id']==best['variant'])
result={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':json.loads((r/'selection_protocol.json').read_text()),'winner':best,'model_request':model,'variant':variant,'candidates':sorted(summary,key=lambda x:x['ecapa'],reverse=True),'test_metrics_seen':False,'listening_verdict':'not_available'}
(r/'selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if 'config' in model:
 model['variants']=[variant]
 (r/'optimized_full_request.json').write_text(json.dumps({'output':str(r/'optimized_full'),'models':[model],'cases':[{'id':c['id'],'source':c['source'],'role':'conversion'} for c in json.loads((r/'cases.json').read_text())]},indent=2))
