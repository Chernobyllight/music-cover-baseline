# -*- coding: utf-8 -*-
import hashlib,json,math,shutil,sys
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import butter,sosfilt
from music_cover.common.paths import ROOT
from music_cover.audio import sha256,decode

EXP=ROOT/'runs/svc_optimization_yang_kun_20261028';EXP.mkdir(exist_ok=True)
RAW=ROOT/'data/raw/references/yang_kun'
manifest=json.loads((RAW/'manifest.json').read_text())
original=json.loads((RAW/'dataset/manifest.json').read_text())
DEST=ROOT/'data/voice_datasets/yang_kun/so_vits_svc/optimization_20261028'
DEST.mkdir(parents=True,exist_ok=True)
rows=[];audit=[];used={}
for track,split in manifest['song_splits'].items():
 if split!='train':continue
 group=manifest['related_song_groups'][track]
 folder=RAW/'sources'/track/'stems';q=json.loads((RAW/'quality'/track/'quality.json').read_text())
 if q['issues']:continue
 lead,sr=sf.read(folder/'lead.wav',dtype='float32',always_2d=True);lead=lead.mean(axis=1)
 vocals,_=sf.read(folder/'vocals.wav',dtype='float32',always_2d=True);vocals=vocals.mean(axis=1)
 n=min(len(lead),len(vocals));lead=lead[:n];vocals=vocals[:n]
 residue=vocals-lead
 ratio=float(20*np.log10(max(np.sqrt(np.mean(residue**2)),1e-9)/max(np.sqrt(np.mean(lead**2)),1e-9)))
 # Avoid a second source-separation pass when the first pass is already solo.
 clean=vocals if ratio < -20 else lead
 pitch=np.load(RAW/'quality'/track/'pitch.npz');f0=pitch['f0'];rms=pitch['rms']
 length=n/sr;start=.15
 while start+4<=length-.1:
  goal=min(start+10,length-.1)
  if length-.1-goal<4:goal=length-.1
  if goal<length-.1:
   lo=int((goal-1)*100);hi=min(len(rms),int((goal+1)*100))
   if hi>lo:goal=(lo+int(np.argmin(rms[lo:hi])))/100
  end=min(length-.1,goal);a,b=round(start*sr),round(end*sr)
  v=f0[round(start*100):round(end*100)];r=rms[round(start*100):round(end*100)]
  active=float(np.mean(r>max(1e-4,np.quantile(r,.9)*.06)))
  voiced=float(np.mean(v>0));duration=(b-a)/sr
  accepted=duration>=4 and active>=.70 and voiced>=.55 and used.get(group,0)+duration<=55
  audit.append({'track':track,'group':group,'start':start,'duration':duration,'active':active,'voiced':voiced,'accepted':accepted,'removed_component_db':ratio})
  if accepted:
   for version,wave in [('expanded_lead',lead),('expanded_clean',clean)]:
    p=DEST/version/'train/yang_kun'/f'{track}_{a:08d}.wav';p.parent.mkdir(parents=True,exist_ok=True)
    x=wave[a:b].copy();x=sosfilt(butter(2,55,fs=sr,btype='highpass',output='sos'),x).astype(np.float32)
    gain=min(10**(6/20),.1/max(np.sqrt(np.mean(x*x)),1e-9),.9/max(abs(x).max(),1e-9));x*=gain
    fade=min(220,len(x)//2);x[:fade]*=np.linspace(0,1,fade);x[-fade:]*=np.linspace(1,0,fade)
    sf.write(p,x,sr,subtype='PCM_16')
    rows.append({'version':version,'path':str(p),'split':'train','group':group,'duration':duration,'source':str(folder/('vocals.wav' if version=='expanded_clean' and ratio < -20 else 'lead.wav')),'start':start,'sha256':sha256(p)})
   used[group]=used.get(group,0)+duration
  start=end
for version in ['expanded_lead','expanded_clean']:
 selected=[r for r in rows if r['version']==version]
 for r in original['clips']:
  if r['split'] not in ('validation','test'):continue
  source=RAW/'dataset'/r['path'];dest=DEST/version/r['split']/'yang_kun'/Path(r['path']).name;dest.parent.mkdir(parents=True,exist_ok=True)
  x,sr=sf.read(source,dtype='float32');sf.write(dest,x,sr,subtype='PCM_16')
  selected.append({'path':str(dest),'split':r['split'],'group':r['group'],'duration':len(x)/sr,'source':str(source),'sha256':sha256(dest)})
 groups={}
 for r in selected:
  if r['group'] in groups:assert groups[r['group']]==r['split']
  groups[r['group']]=r['split']
 for split in ['train','validation','test']:(DEST/version/(split+'.txt')).write_text(''.join(r['path']+'\n' for r in selected if r['split']==split))
 summary={s:{'clips':sum(r['split']==s for r in selected),'seconds':sum(r['duration'] for r in selected if r['split']==s),'groups':len({r['group'] for r in selected if r['split']==s})} for s in ['train','validation','test']}
 (DEST/version/'manifest.json').write_text(json.dumps({'clips':selected,'summary':summary,'split_policy':'same song groups as original; prompt excluded; validation/test unchanged recordings','voice_id':'yang_kun'},ensure_ascii=False,indent=2))
 print(version,summary,flush=True)
(DEST/'screening.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))

mapping=[('hero_celtic','lhd_bu_bai_de_ying_xiong_celtic_epic_yang_kun_finetuned_20261028','selection'),('bad_girl_funk','lhd_huai_nv_hai_funk_soul_yang_kun_finetuned_20261027','selection'),('hero_blues','lhd_bu_bai_de_ying_xiong_blues_rock_yang_kun_finetuned_20261028','test'),('autumn_bossa','lhd_voice_candidates_validation','test')]
cases=[]
for name,run_name,split in mapping:
 run=ROOT/'runs'/run_name;state=json.loads((run/'stage_state.json').read_text());cfg=json.loads((run/'config.resolved.json').read_text())
 item=state['stages']['04_vocal_separation']['outputs']['vocals'];p=Path(item['path']);assert sha256(p)==item['sha256']
 out=EXP/'cases'/name;out.mkdir(parents=True,exist_ok=True)
 decode(p,out/'source.wav',rate=44100,channels=1)
 for i,(start,duration) in enumerate([(15,18),(65,18)]):decode(out/'source.wav',out/f'window_{i}.wav',rate=44100,channels=1,start=start,duration=duration)
 cases.append({'id':name,'split':split,'source_run':str(run),'source':str(out/'source.wav'),'source_sha256':sha256(out/'source.wav'),'windows':[str(out/f'window_{i}.wav') for i in range(2)],'request':cfg['request'],'note':'Existing styled vocals; same input for all models. Autumn case deliberately retains higher female register.'})
(EXP/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
print('prepared cases',len(cases))
