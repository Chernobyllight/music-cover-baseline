import json, os, shutil
from pathlib import Path
from music_cover.common.paths import ROOT
from music_cover.backends.svc_train import train_model

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
base=ROOT/'runs/voice_training/yang_kun_so_vits_svc_v001_20261028'
run=ROOT/'runs/svc_optimization_yang_kun_20261028/training/original_long'
run.mkdir(parents=True,exist_ok=True)
(run/'checkpoints').mkdir(exist_ok=True)
shutil.copy2(base/'model_config.json',run/'model_config.json')
original=base/'checkpoints/step_00001000.pth'
target=run/'checkpoints'/original.name
if not target.exists():shutil.copy2(original,target)
c=json.loads((ROOT/'configs/voice_training/so_vits_svc/yang_kun.yaml').read_text())
c['training'].update(max_steps=12000,save_every=3000)
out=run/'report';out.mkdir(exist_ok=True)
manifest=json.loads((ROOT/c['dataset_dir']/'manifest.json').read_text())
train_model(c,run,out,manifest,resume=True)
