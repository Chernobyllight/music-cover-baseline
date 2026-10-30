import json,copy
from pathlib import Path
from music_cover.common.paths import ROOT
r=ROOT/'runs/svc_optimization_yang_kun_20261028'
initial=json.loads((r/'screening_initial_request.json').read_text())
cases=[]
for c in json.loads((r/'cases.json').read_text()):
 if c['split']=='selection':
  for i,w in enumerate(c['windows']):cases.append({'id':c['id']+'_w'+str(i),'source':w,'role':'conversion'})
for p in sorted((ROOT/'data/raw/references/yang_kun/dataset/validation').glob('*.wav')):cases.append({'id':'self_'+p.stem,'source':str(p),'role':'reconstruction'})
models=[]
b=copy.deepcopy(initial['models'][0]);b['variants']=[{'id':'default','settings':{}}];models.append(b)
for name,step in [('clean_vec768',4000),('clean_vec256',6000),('clean_vec256',8000),('low_lr_refine',2000),('long_segment_refine',2000),('long_segment_refine',4000)]:
 t=r/'training'/name
 models.append({'id':name+'_'+str(step),'checkpoint':str(t/'checkpoints'/f'step_{step:08d}.pth'),'config':str(t/'model_config.json'),'variants':[{'id':'chunk8','settings':{'noise':.1,'chunk':8.,'overlap':.2}}]})
for n,items in [('a',models[:3]),('b',models[3:])]:
 (r/f'final_validation_{n}_request.json').write_text(json.dumps({'output':str(r/('final_validation_'+n)),'models':items,'cases':cases},indent=2))
(r/'validation_cases.json').write_text(json.dumps(cases,indent=2))
# Freeze this decision before any final held-out scoring.
(r/'selection_protocol.json').write_text(json.dumps({'conversion_cases':[c['id'] for c in cases if c['role']=='conversion'],'reconstruction_cases':[c['id'] for c in cases if c['role']=='reconstruction'],'eligibility':{'f0_within_one_semitone_each_conversion_min':.9,'paired_reconstruction_mean_max_baseline_ratio':1.05,'finite_unclipped_nonsilent':True},'ranking':'Maximum mean ECAPA on eligible validation conversion clips; fidelity and F0 shown alongside; no perceptual superiority claim.','test_cases':['hero_blues','autumn_bossa'],'test_reference_split':'test','subjective_quality':'not yet listened by human'},indent=2))
