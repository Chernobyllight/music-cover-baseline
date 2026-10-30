import json
from pathlib import Path
from music_cover.common.paths import ROOT
exp=ROOT/'runs/svc_optimization_yang_kun_20261028';initial=json.loads((exp/'screening_initial_request.json').read_text());base=initial['models'][0]
variants=[{'id':'enhance_'+str(k),'settings':{'noise':.1,'enhance':True,'enhancer_key':k}} for k in [0,6,12]]
variants += [{'id':'f0_fcpe','settings':{'noise':.1,'f0':'fcpe'}}]
models=[{**base,'variants':variants}]
for steps in [3000,6000]:
 run=exp/'training/original_long'
 models.append({'id':'original_'+str(steps),'checkpoint':str(run/'checkpoints'/f'step_{steps:08d}.pth'),'config':str(run/'model_config.json'),'variants':[{'id':'noise_01','settings':{'noise':.1}},{'id':'enhance','settings':{'noise':.1,'enhance':True}},{'id':'chunk8','settings':{'noise':.1,'chunk':8.,'overlap':.2}}]})
(exp/'screening_followups_request.json').write_text(json.dumps({'output':str(exp/'screening_followups'),'models':models,'cases':initial['cases']},indent=2))
for kind in ['shallow','full']:
 models=[]
 for steps in [2000,6000]:
  variants=[{'id':'k_'+str(k),'k_step':k,'initial_outputs':str(exp/'screening_initial/svc_1000/noise_0.1')} for k in [50,100,200]]
  if kind=='full':variants=[{'id':'pure','k_step':1000}]
  models.append({'id':kind+'_'+str(steps),'checkpoint':str(exp/'training'/('diffusion_'+kind)/'checkpoints'/f'step_{steps:08d}.pt'),'variants':variants})
 (exp/('screening_diffusion_'+kind+'_request.json')).write_text(json.dumps({'output':str(exp/('screening_diffusion_'+kind)),'models':models,'cases':initial['cases']},indent=2))
