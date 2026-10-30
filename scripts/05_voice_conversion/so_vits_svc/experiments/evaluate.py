"""Frozen ECAPA/F0 evaluation plus paired self-reconstruction diagnostics."""
import argparse,json,hashlib,shutil,traceback
from pathlib import Path
import numpy as np
from scipy.signal import stft
from music_cover.common.paths import ROOT
from music_cover.backends.voice_metrics import VoiceMetrics
from music_cover.audio import read_at,estimate_alignment,shift_audio
from music_cover.pitch_selection import load_pitch

p=argparse.ArgumentParser();p.add_argument('--outputs',required=True);p.add_argument('--split',default='validation',choices=['validation','test']);a=p.parse_args()
path=Path(a.outputs);rows=json.loads(path.read_text());out=path.parent/'evaluation';out.mkdir(exist_ok=True)
refs=[str(x) for x in sorted((ROOT/'data/raw/references/yang_kun/dataset'/a.split).glob('*.wav'))]
engine=VoiceMetrics();original_pitch=engine.pitch;cache=ROOT/'runs/svc_optimization_yang_kun_20261028/pitch_cache';cache.mkdir(exist_ok=True)
def pitch_cached(y,target):
 key=hashlib.sha256(y.tobytes()).hexdigest();f=cache/(key+'.npz')
 if not f.exists():original_pitch(y,f)
 shutil.copy2(f,target);return load_pitch(target)
engine.pitch=pitch_cached
results=[]
for row in rows:
 if row['status']!='succeeded':continue
 folder=out/row['model']/row['variant']/row['case'];folder.mkdir(parents=True,exist_ok=True)
 f=folder/'result.json'
 if f.exists():
  result=json.loads(f.read_text())
  if result.get('evaluation_split')!=a.split:raise ValueError('Cached evaluation split differs; use a separate output folder')
 else:
  try:
   metrics=engine.compare(row['source'],row['output'],refs,folder)
   x=read_at(row['source'],16000);y=read_at(row['output'],16000);al=estimate_alignment(x,y,16000);y=shift_audio(y,al['applied_shift_samples'],len(x));x=x.mean(axis=1);y=y.mean(axis=1)
   mx=np.sqrt(np.mean(x*x));my=np.sqrt(np.mean(y*y));yn=y*mx/max(my,1e-9)
   spectral=[]
   for size in [512,1024,2048]:
    sx=abs(stft(x,16000,nperseg=size,noverlap=size*3//4)[2]);sy=abs(stft(yn,16000,nperseg=size,noverlap=size*3//4)[2])
    spectral.append(float(np.mean(abs(np.log(np.maximum(sx,1e-5))-np.log(np.maximum(sy,1e-5))))))
   # Meaningful paired fidelity only for target singer identity reconstruction.
   paired={'multi_resolution_log_spectral_l1':float(np.mean(spectral))} if row.get('role')=='reconstruction' else None
   result={**{k:v for k,v in row.items() if k!='report'},'metrics':metrics,'paired_reconstruction':paired,'evaluation_split':a.split}
   f.write_text(json.dumps(result,indent=2))
  except Exception as e:
   result={**row,'status':'evaluation_failed','error':str(e)};(folder/'error.txt').write_text(traceback.format_exc())
 results.append(result)
 (out/'results.json').write_text(json.dumps(results,indent=2))
 print(json.dumps({'model':row['model'],'variant':row['variant'],'case':row['case'],'status':result['status'],'ecapa':result.get('metrics',{}).get('speaker_similarity'),'pitch':result.get('metrics',{}).get('pitch',{}).get('within_one_semitone_fraction'),'reconstruction':result.get('paired_reconstruction')}),flush=True)
