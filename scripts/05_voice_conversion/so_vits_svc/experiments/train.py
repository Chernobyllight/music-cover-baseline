import argparse,json,shutil,hashlib
from pathlib import Path
from core import ROOT,EXP,ASSETS,runtime,prepare_features,build
from training import train_model
import torch
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--encoder',default='vec768l12');p.add_argument('--data',default='expanded_clean');p.add_argument('--steps',type=int,default=6000);p.add_argument('--volume',action='store_true');p.add_argument('--snake',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--segment-size',type=int,default=10240);p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--warmstart');a=p.parse_args()
run=EXP/'training'/a.name;run.mkdir(parents=True,exist_ok=True)
origin=ROOT/'data/voice_datasets/yang_kun/so_vits_svc/optimization_20261028'/a.data
dataset=origin.with_name(a.data+'_'+a.encoder)
if not dataset.exists():
 dataset.mkdir(parents=True)
 for src in origin.glob('*/yang_kun/*.wav'):
  dest=dataset/src.relative_to(origin);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
 for split in ['train','validation','test']:(dataset/(split+'.txt')).write_text(''.join(str(f)+'\n' for f in sorted((dataset/split).rglob('*.wav'))))
 shutil.copy2(origin/'manifest.json',dataset/'source_manifest.json')
with runtime(run/'runtime'):
 prepare_features(dataset,a.encoder)
 base={'vec768l12':ROOT/'models/SoVitsSVC/G_0.pth','vec256l9':ASSETS/'vec256l9_G.pth','hubertsoft':ASSETS/'hubertsoft_G.pth','whisper-ppg':ASSETS/'whisper_ppg_G.pth'}[a.encoder]
 d=torch.load(base,map_location='cpu',weights_only=True)['model']
 config=json.loads((ROOT/'configs/voice_training/so_vits_svc/model_template.json').read_text())
 config['model'].update(speech_encoder=a.encoder,ssl_dim=int(d['pre.weight'].shape[1]),gin_channels=int(d['dec.cond.weight'].shape[1]),vol_embedding=a.volume,vocoder_name='nsf-snake-hifigan' if a.snake else 'nsf-hifigan')
 config['model']['filter_channels']=int(d['enc_p.enc_.ffn_layers.0.conv_1.weight'].shape[0])
 config['experiment_base']=str(base)
 if a.warmstart:config['experiment_base']=str((ROOT/a.warmstart).resolve());config['experiment_warmstart']=True
 config['train'].update(vol_aug=a.volume,segment_size=a.segment_size,learning_rate=a.learning_rate)
 config['data'].update(training_files=str(dataset/'train.txt'),validation_files=str(dataset/'validation.txt'))
 config['experiment_data_manifest_sha256']=hashlib.sha256((dataset/'source_manifest.json').read_bytes()).hexdigest()
 g,coverage=build(config,base,replace_speaker=True);del g;torch.cuda.empty_cache()
 (run/'loading.json').write_text(json.dumps(coverage,indent=2))
 (run/'model_config.json').write_text(json.dumps(config,indent=2))
 from modules import mel_processing
 mel_processing.hann_window.clear();mel_processing.mel_basis.clear()
c={'dataset_dir':str(dataset),'training':{'learning_rate':a.learning_rate,'batch_size':6,'max_steps':a.steps,'save_every':2000,'seed':42}}
manifest={'identity':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()}
out=run/'report';out.mkdir(exist_ok=True)
train_model(c,run,out,manifest,resume=a.resume)
