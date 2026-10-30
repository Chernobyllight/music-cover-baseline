# -*- coding: utf-8 -*-
import json,shutil
from pathlib import Path
import numpy as np,soundfile as sf,librosa
from scipy.signal import butter,sosfilt
from core import ROOT,EXP,runtime
from music_cover.audio import sha256
base=ROOT/'data/voice_datasets/yang_kun/so_vits_svc/optimization_20261028'
dest=base/'expanded_full';dest.mkdir(exist_ok=True);origin=base/'expanded_clean'
manifest=json.loads((origin/'manifest.json').read_text());rows=[]
for r in manifest['clips']:
 if r['split']=='train' and r['group']=='yang_kun:空城':continue
 src=Path(r['path']);p=dest/src.relative_to(origin);p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
 rows.append({**r,'path':str(p),'sha256':sha256(p)})
lead,sr=sf.read(base/'extra_kongcheng/lead.wav',dtype='float32',always_2d=True);lead=lead.mean(axis=1)
vocals,_=sf.read(base/'extra_kongcheng/vocals.wav',dtype='float32',always_2d=True);vocals=vocals.mean(axis=1)
with runtime(EXP/'data_extension_runtime'):
 import utils
 pred=utils.get_f0_predictor('rmvpe',hop_length=512,sampling_rate=44100,device='cuda',threshold=.05)
 used=0
 for start in np.arange(.2,len(lead)/sr-5,10):
  end=min(start+10,len(lead)/sr);x=lead[round(start*sr):round(end*sr)]
  f0,uv=pred.compute_f0_uv(x);r=librosa.feature.rms(y=x,frame_length=1024,hop_length=512)[0]
  active=float(np.mean(r>max(1e-4,np.quantile(r,.9)*.08)))
  if np.mean(uv)<.6 or active<.75 or used+len(x)/sr>190:continue
  v=vocals[round(start*sr):round(end*sr)];ratio=20*np.log10(max(np.sqrt(np.mean((v-x)**2)),1e-9)/max(np.sqrt(np.mean(x*x)),1e-9))
  if ratio < -20:x=v
  x=sosfilt(butter(2,55,fs=sr,btype='highpass',output='sos'),x).astype(np.float32)
  x*=min(2.,.1/max(np.sqrt(np.mean(x*x)),1e-9),.9/max(abs(x).max(),1e-9));x[:220]*=np.linspace(0,1,220);x[-220:]*=np.linspace(1,0,220)
  p=dest/'train/yang_kun'/f'full_kongcheng_{int(start*100):05d}.wav';p.parent.mkdir(parents=True,exist_ok=True);sf.write(p,x,sr,subtype='PCM_16')
  rows.append({'path':str(p),'split':'train','group':'yang_kun:空城','duration':len(x)/sr,'source':str(base/'extra_kongcheng/lead.wav'),'start':float(start),'voiced_fraction':float(np.mean(uv)),'active_fraction':active,'sha256':sha256(p)})
  used+=len(x)/sr
for split in ['train','validation','test']:(dest/(split+'.txt')).write_text(''.join(r['path']+'\n' for r in rows if r['split']==split))
summary={s:{'clips':sum(r['split']==s for r in rows),'seconds':sum(r['duration'] for r in rows if r['split']==s)} for s in ['train','validation','test']}
(dest/'manifest.json').write_text(json.dumps({'clips':rows,'summary':summary,'split_policy':manifest['split_policy'],'voice_id':'yang_kun'},ensure_ascii=False,indent=2));print(summary,flush=True)
