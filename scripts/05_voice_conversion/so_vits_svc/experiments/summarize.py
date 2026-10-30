"""Audit generated/evaluated counts and make a compact, reproducible report."""
import json,collections,statistics
from pathlib import Path
from music_cover.common.paths import ROOT
from music_cover.audio import sha256
r=ROOT/'runs/svc_optimization_yang_kun_20261028'
screen=[];audit=[]
for reqfile in sorted(r.glob('screening_*_request.json')):
 req=json.loads(reqfile.read_text());folder=Path(req['output']);expected=len(req['cases'])*sum(len(m.get('variants',[{}])) for m in req['models']);outputs=json.loads((folder/'outputs.json').read_text());rows=json.loads((folder/'evaluation/results.json').read_text())
 assert len(outputs)==expected and all(x['status']=='succeeded' for x in outputs),(folder,expected,len(outputs))
 assert len(rows)==expected and all(x['status']=='succeeded' for x in rows),(folder,len(rows))
 audit.append({'experiment':folder.name,'generated':len(outputs),'evaluated':len(rows)})
 groups=collections.defaultdict(list)
 for x in rows:groups[(x['model'],x['variant'])].append(x)
 for (model,variant),xs in groups.items():
  c=[x for x in xs if x['role']=='conversion'];s=[x for x in xs if x['role']=='reconstruction']
  screen.append({'experiment':folder.name,'model':model,'variant':variant,'ecapa':statistics.mean(x['metrics']['speaker_similarity'] for x in c),'f0':statistics.mean(x['metrics']['pitch'].get('within_one_semitone_fraction',0) or 0 for x in c),'self_logspec':statistics.mean(x['paired_reconstruction']['multi_resolution_log_spectral_l1'] for x in s)})
final=[]
for folder,method in [('baseline_full','svc_original'),('optimized_full','svc_optimized'),('seed_full','seed_1000')]:
 rows=json.loads((r/folder/'evaluation/results.json').read_text());assert len(rows)==4 and all(x['status']=='succeeded' and x['evaluation_split']=='test' for x in rows)
 for x in rows:
  final.append({'case':x['case'],'method':method,'ecapa':x['metrics']['speaker_similarity'],'f0':x['metrics']['pitch']['within_one_semitone_fraction'],'f0_coverage':x['metrics']['pitch']['source_coverage'],'alignment':x['metrics']['alignment'],'audio':x['metrics']['audio'],'output':x['output']})
cases=json.loads((r/'cases.json').read_text())
for c in cases:assert sha256(c['source'])==c['source_sha256']
# Existing exported models are content-addressed; check their files were not replaced.
unchanged=[]
for path in [ROOT/'models/voices/yang_kun/so_vits_svc/v001_20261028/voice.json',ROOT/'models/voices/yang_kun/seed_vc/catalog_v001_20261027/voice.json']:
 d=json.loads(path.read_text())
 for key in ['checkpoint','config','validation']:
  if isinstance(d.get(key),dict) and 'sha256' in d[key]:assert sha256(path.parent/d[key]['path'])==d[key]['sha256']
 unchanged.append({'profile':str(path),'sha256':sha256(path),'assets_match':True})
training=[]
for p in sorted((r/'training').iterdir()):
 t=p/'report/training.json'
 if not t.exists():t=p/'training.json'
 training.append({'name':p.name,'report':json.loads(t.read_text()) if t.exists() else None,'early_stop':json.loads((p/'early_stop.json').read_text()) if (p/'early_stop.json').exists() else None})
result={'selection':json.loads((r/'selection.json').read_text()),'cases':cases,'screening':screen,'screening_audit':audit,'final':final,'existing_profiles':unchanged,'training':training,'contentvec_equivalence':json.loads((r/'contentvec_equivalence.json').read_text()),'limitations':['Single target singer; 3 songs and 4 arrangements, 2 case arrangements held out from selection.','One fixed inference seed; no repeated-seed confidence intervals.','ECAPA measures embedding similarity, not a calibrated perceptual quality/MOS score.','Training datasets and update counts differ across complete systems.','No human listening verdict has been collected.']}
(r/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('verified screening files',sum(x['generated'] for x in audit),'final',len(final))
