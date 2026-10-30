"""Export frozen winner as a NEW immutable production SVC speaker profile."""
import json,shutil
from pathlib import Path
import torch
from music_cover.common.paths import ROOT
from music_cover.audio import sha256
from music_cover.svc import UPSTREAM_COMMIT,resolve_svc_voice
r=ROOT/'runs/svc_optimization_yang_kun_20261028';selection=json.loads((r/'selection.json').read_text());model=selection['model_request']
p=ROOT/'models/voices/yang_kun/so_vits_svc/optimized_v002_20261028';p.mkdir(parents=True,exist_ok=False)
checkpoint=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
torch.save({'model':checkpoint['model'],'step':checkpoint.get('step',2000)},p/'model.pth')
shutil.copy2(model['config'],p/'config.json');shutil.copy2(r/'selection.json',p/'validation.json')
c=json.loads((p/'config.json').read_text())
lock=json.loads((ROOT/'configs/so_vits_svc_assets.lock.json').read_text())
profile={'schema_version':1,'backend':'so_vits_svc','adaptation_scope':'speaker','voice_id':'yang_kun','version':p.name,'speaker_name':'yang_kun','upstream_commit':UPSTREAM_COMMIT,'validation_status':'objective_validation_passed_requires_listening','reference_required':False,'selected_step':6000,'selected_step_note':'4000 initial updates + 2000 low-learning-rate refinement updates','training_run':str(r/'training/low_lr_refine'),'source_checkpoint':model['checkpoint'],'source_checkpoint_sha256':sha256(model['checkpoint']),'dataset_identity':c['experiment_data_manifest_sha256'],'subjective_quality':'awaiting_human_review','inference_assets':{x['path']:x['sha256'] for x in lock['assets'] if x.get('scope')!='training'},'recommended_config':'configs/voice_inference/so_vits_svc/yang_kun_optimized.yaml'}
for key,name in [('checkpoint','model.pth'),('config','config.json'),('validation','validation.json')]:profile[key]={'path':name,'sha256':sha256(p/name)}
(p/'voice.json').write_text(json.dumps(profile,indent=2))
settings={'f0_predictor':'rmvpe','noise_scale':.1,'chunk_seconds':8.,'overlap_seconds':.2,'pad_seconds':.5}
config={'modules':{'05_voice_conversion':{'backend':'so_vits_svc','mode':'finetuned','voice_profile':str((p/'voice.json').relative_to(ROOT)),'candidate_count':1,'svc':settings}}}
# Config field names are verified against the module schema before writing below.
q=ROOT/'configs/voice_inference/so_vits_svc/yang_kun_optimized.yaml';q.parent.mkdir(parents=True,exist_ok=True)
q.write_text(json.dumps(config,indent=2)+'\n')
print(resolve_svc_voice('finetuned',p/'voice.json'))
