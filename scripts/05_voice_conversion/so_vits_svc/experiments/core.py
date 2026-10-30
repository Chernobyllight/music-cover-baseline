"""Isolated SVC experimental backends. No changes to Seed-VC or default SVC."""
import contextlib,json,os,pickle,sys,time,types
from pathlib import Path
import numpy as np,soundfile as sf,torch,librosa
from music_cover.common.paths import ROOT
from music_cover.audio import sha256
from music_cover.backends.so_vits_svc import runtime as original_runtime
ASSETS=ROOT/'models/SoVitsSVC/optimization_20261028'
EXP=ROOT/'runs/svc_optimization_yang_kun_20261028'
UPSTREAM=ROOT/'third_party/so-vits-svc'
torch.set_num_threads(4)

@contextlib.contextmanager
def runtime(folder):
 with original_runtime(folder):
  p=Path('pretrain/fcpe.pt')
  if (ASSETS/'fcpe.pt').exists() and not p.exists():p.symlink_to(ASSETS/'fcpe.pt')
  try:yield
  finally:
   if p.is_symlink():p.unlink()

class Encoder:
 def __init__(self,name):
  self.name=name
  if name in ('vec768l12','vec256l9'):
   from music_cover.backends.svc_contentvec import ContentVec,_ContentVecUnpickler
   self.impl=ContentVec()
   if name=='vec256l9':
    mod=types.ModuleType('legacy_pickle');mod.Unpickler=_ContentVecUnpickler;mod.load=pickle.load
    p=torch.load(ROOT/'models/SoVitsSVC/contentvec.pt',map_location='cpu',pickle_module=mod,weights_only=False)['model']
    self.proj=torch.nn.Linear(768,256).cuda().eval().requires_grad_(False)
    self.proj.load_state_dict({'weight':p['final_proj.weight'].float(),'bias':p['final_proj.bias'].float()})
  elif name=='hubertsoft':
   from vencoder.HubertSoft import HubertSoft
   self.impl=HubertSoft(str(ASSETS/'hubert-soft.pt'),'cuda');self.impl.model.eval().requires_grad_(False)
  elif name=='whisper-ppg':
   from vencoder.WhisperPPG import WhisperPPG
   self.impl=WhisperPPG(str(ASSETS/'medium.pt'),'cuda');self.impl.model.eval().requires_grad_(False)
  else:raise ValueError(name)
 def encoder(self,wave):
  with torch.inference_mode():
   if self.name=='vec256l9':
    h=self.impl.model(wave.reshape(1,-1),output_hidden_states=True).hidden_states[9]
    return self.proj(h).transpose(1,2)
   return self.impl.encoder(wave)

def build(config,checkpoint,device='cuda',replace_speaker=False):
 from models import SynthesizerTrn
 g=SynthesizerTrn(config['data']['filter_length']//2+1,config['train']['segment_size']//512,**config['model'])
 actual=Path(config.get('experiment_base',checkpoint)) if replace_speaker else Path(checkpoint)
 if config.get('experiment_warmstart') and replace_speaker:replace_speaker=False
 state=torch.load(actual,map_location='cpu',weights_only=False)['model'];current=g.state_dict()
 allowed={'emb_g.weight'} if replace_speaker else set()
 if replace_speaker and config['model'].get('vol_embedding'):allowed|={'emb_vol.weight','emb_vol.bias'}
 if replace_speaker and config['model']['vocoder_name']=='nsf-snake-hifigan':
  allowed|={k for k in current if 'activation' in k or 'alpha' in k or 'beta' in k or k.startswith('dec.snake')}
 missing=set(current)-set(state);extra=set(state)-set(current)
 mismatch=[k for k in current.keys()&state.keys() if current[k].shape!=state[k].shape]
 if missing-allowed or extra or set(mismatch)-allowed:raise ValueError({'missing':sorted(missing-allowed),'extra':sorted(extra),'mismatch':mismatch})
 for k in allowed:
  if k in current:state[k]=current[k]
 g.load_state_dict(state,strict=True)
 return g.to(device),{'base':str(actual),'loaded':len(current)-len(allowed),'initialized':sorted(allowed),'sha256':sha256(actual)}

def prepare_features(dataset,encoder_name):
 import utils
 from modules.mel_processing import spectrogram_torch
 encoder=Encoder(encoder_name);pitch=utils.get_f0_predictor('rmvpe',hop_length=512,sampling_rate=44100,device='cuda',threshold=.05)
 for file in sorted(Path(dataset).glob('*/yang_kun/*.wav')):
  if Path(str(file)+'.soft.pt').exists() and file.with_suffix('.spec.pt').exists():continue
  x,sr=sf.read(file,dtype='float32');assert sr==44100 and x.ndim==1
  with torch.inference_mode():
   c=encoder.encoder(torch.from_numpy(librosa.resample(x,orig_sr=sr,target_sr=16000)).cuda())
   f0,uv=pitch.compute_f0_uv(x)
   spec=spectrogram_torch(torch.from_numpy(x).cuda()[None],2048,44100,512,2048,center=False)[0]
   volume=utils.Volume_Extractor(512).extract(torch.from_numpy(x)[None]).numpy()
  torch.save(c.cpu(),str(file)+'.soft.pt');torch.save(spec.cpu(),file.with_suffix('.spec.pt'))
  np.save(str(file)+'.f0.npy',np.stack([f0,uv]));np.save(str(file)+'.vol.npy',volume)
  print('FEATURE',file,flush=True)

class Converter:
 def __init__(self,checkpoint,config):
  import utils
  self.utils=utils;self.config=config;self.model,self.coverage=build(config,checkpoint)
  self.model.eval().requires_grad_(False);self.encoder=Encoder(config['model']['speech_encoder']);self.pitches={};self.enhancer=None
  self.sid=torch.zeros((1,1),device='cuda',dtype=torch.long);self.index=None;self.bank=None;self.diffusion=None
 def infer(self,source,destination,options=None):
  p={'noise':.4,'f0':'rmvpe','retrieval':0.,'index':None,'enhance':False,'enhancer_key':0,'rms_mix':1.,'chunk':20.,'overlap':.5,'pad':.5,'seed':42,**(options or {})}
  if p['f0'] not in self.pitches:self.pitches[p['f0']]=self.utils.get_f0_predictor(p['f0'],hop_length=512,sampling_rate=44100,device='cuda',threshold=.05)
  pitch=self.pitches[p['f0']]
  if p['enhance'] and self.enhancer is None:
   from modules.enhancer import Enhancer
   self.enhancer=Enhancer('nsf-hifigan',str(next((ASSETS/'nsf_hifigan').rglob('model'))),'cuda')
  if p['retrieval'] and (self.index is None or self.index_path!=p['index']):
   import faiss
   self.index=faiss.read_index(p['index']);self.bank=self.index.reconstruct_n(0,self.index.ntotal);self.index_path=p['index']
  x,sr=sf.read(source,dtype='float32',always_2d=True);x=x.mean(axis=1);assert sr==44100
  width=round(p['chunk']*sr);overlap=round(p['overlap']*sr);pad=round(p['pad']*sr)
  result=np.zeros(len(x),np.float32);weights=np.zeros(len(x),np.float32);segments=[];start=time.monotonic()
  t=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None]
  with torch.inference_mode():
   for j,first in enumerate(range(0,len(x),width-overlap)):
    last=min(len(x),first+width);core=x[first:last]
    if np.sqrt(np.mean(core**2))<1e-5:y=np.zeros(len(core),np.float32)
    else:
     block=np.pad(x[max(0,first-pad):min(len(x),last+pad)],(max(0,pad-first),max(0,last+pad-len(x))))
     f0,uv=pitch.compute_f0_uv(block)
     c=self.encoder.encoder(torch.from_numpy(librosa.resample(block,orig_sr=sr,target_sr=16000)).cuda())
     c=self.utils.repeat_expand_2d(c.squeeze(0),len(f0),mode=self.config['data']['unit_interpolate_mode'])[None]
     if p['retrieval']:
      a=np.ascontiguousarray(c[0].T.cpu().numpy());distance,idx=self.index.search(a,8)
      w=1/np.maximum(distance,1e-6)**2;w/=w.sum(axis=1,keepdims=True)
      recalled=np.sum(self.bank[idx]*w[:,:,None],axis=1)
      c=t(p['retrieval']*recalled+(1-p['retrieval'])*a).transpose(1,2)
     volume=self.utils.Volume_Extractor(512).extract(t(block))[None].cuda() if self.config['model'].get('vol_embedding') else None
     y,_=self.model.infer(c,f0=t(f0),uv=t(uv),g=self.sid,noice_scale=p['noise'],seed=p['seed']+j,predict_f0=False,vol=volume)
     y=y[0,0]
     if p['enhance']:y,_=self.enhancer.enhance(y[None],sr,t(f0)[:,:,None],512,adaptive_key=p['enhancer_key'])
     if p['rms_mix']!=1:y=self.utils.change_rms(block,sr,y,sr,p['rms_mix'])
     y=y.float().cpu().numpy();y=np.pad(y,(0,max(0,len(block)-len(y))))[pad:pad+len(core)]
    w=np.ones(len(core),np.float32);n=min(overlap,len(core))
    if first:w[:n]*=np.linspace(1e-6,1,n)
    if last<len(x):w[-n:]*=np.linspace(1,1e-6,n)
    result[first:last]+=y*w;weights[first:last]+=w;segments.append([first,last])
    if last==len(x):break
  result/=np.maximum(weights,1e-12);assert np.isfinite(result).all()
  peak=float(np.max(np.abs(result)));gain=min(1.,.99/max(peak,1e-9));dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
  sf.write(dest,result*gain,sr,subtype='FLOAT')
  return {'options':p,'seconds':len(x)/sr,'wall_seconds':time.monotonic()-start,'peak':peak,'gain':gain,'segments':segments,'loading':self.coverage}
