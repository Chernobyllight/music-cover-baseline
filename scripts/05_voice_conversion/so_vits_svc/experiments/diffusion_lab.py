"""Train official SVC Unit2Mel; evaluate both shallow and full diffusion."""
import argparse,json,time,random,hashlib
from pathlib import Path
from core import ROOT,EXP,ASSETS,runtime,Encoder,Converter
import numpy as np,soundfile as sf,torch,librosa

p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','infer']);p.add_argument('--kind',choices=['full','shallow'],default='shallow');p.add_argument('--steps',type=int,default=6000);p.add_argument('--request');p.add_argument('--resume',action='store_true');a=p.parse_args()
if a.request:a.request=str(Path(a.request).resolve())
run=EXP/'training'/('diffusion_'+a.kind);run.mkdir(parents=True,exist_ok=True)
with runtime(run/'runtime'):
 from diffusion.unit2mel import Unit2Mel
 from diffusion.vocoder import Vocoder
 import utils
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 vocoder=Vocoder('nsf-hifigan',str(next((ASSETS/'nsf_hifigan').rglob('model'))),'cuda')
 def model():return Unit2Mel(768,1,True,128,20,512,256,1000,200 if a.kind=='shallow' else 1000).cuda()
 if a.mode=='train':
  net=model();payload=torch.load(ASSETS/('diffusion_'+a.kind+'.pt'),map_location='cpu',weights_only=False);net.load_state_dict(payload['model'],strict=True);del payload
  optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=0)
  dataset=ROOT/'data/voice_datasets/yang_kun/so_vits_svc/optimization_20261028/expanded_clean_vec768l12'
  data=[]
  for f in sorted((dataset/'train').rglob('*.wav')):
   x,sr=sf.read(f,dtype='float32');mel_path=Path(str(f)+'.diffmel.pt')
   if not mel_path.exists():
    with torch.no_grad():mel=vocoder.extract(torch.from_numpy(x).cuda()[None],sr)[0].cpu();torch.save(mel,mel_path)
   mel=torch.load(mel_path,weights_only=True).cuda()
   f0,uv=np.load(str(f)+'.f0.npy');n=min(len(f0),len(mel));units=torch.load(str(f)+'.soft.pt',weights_only=True).squeeze(0).cuda()
   units=utils.repeat_expand_2d(units,len(f0),'nearest').T[:n]
   vol=np.load(str(f)+'.vol.npy')[:n];n=min(n,len(vol))
   data.append((units[:n],torch.tensor(f0[:n],device='cuda',dtype=torch.float32)[:,None],torch.tensor(vol[:n],device='cuda',dtype=torch.float32)[:,None],mel[:n]))
  print('dataset',len(data),flush=True);net.train();start=time.monotonic();history=[]
  ckpts=run/'checkpoints';ckpts.mkdir(exist_ok=True)
  existing=sorted(ckpts.glob('step_*.pt'));initial=0
  if existing and not a.resume:raise ValueError('Existing diffusion run; explicitly resume')
  if a.resume:
   old=json.loads((run/'training.json').read_text())
   if old['dataset']!=str(dataset) or old['kind']!=a.kind:raise ValueError('Diffusion resume identity mismatch')
   saved=torch.load(existing[-1],map_location='cpu',weights_only=False);net.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);initial=saved['step']
   random.setstate(saved['rng_python']);torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state_all(saved['rng_cuda']);del saved
   history=json.loads((run/'losses.json').read_text())
  for step in range(initial+1,a.steps+1):
   samples=[]
   for _ in range(12):
    row=random.choice(data);left=random.randrange(max(1,len(row[0])-172));samples.append(tuple(t[left:left+172] for t in row))
   u,f,v,m=[torch.stack(x) for x in zip(*samples)]
   optimizer.zero_grad(set_to_none=True);loss=net(u,f,v,gt_spec=m,infer=False,k_step=200 if a.kind=='shallow' else 1000)
   if not torch.isfinite(loss):raise ValueError('Nonfinite diffusion loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True);optimizer.step()
   for group in optimizer.param_groups:group['lr']=1e-4*.5**(step/10000)
   if step%50==0 or step==1:
    r={'step':step,'loss':float(loss.detach()),'wall_seconds':time.monotonic()-start};history.append(r);print(json.dumps(r),flush=True)
    (run/'losses.json').write_text(json.dumps(history))
   if step%2000==0 or step==a.steps:
    torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'rng_python':random.getstate(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()},ckpts/f'step_{step:08d}.pt')
  (run/'training.json').write_text(json.dumps({'steps':a.steps,'kind':a.kind,'base':str(ASSETS/('diffusion_'+a.kind+'.pt')),'dataset':str(dataset),'seconds':time.monotonic()-start,'train_clips':len(data),'settings':{'batch':12,'lr':1e-4,'frames':172,'layers':20,'channels':512,'k_step_max':200 if a.kind=='shallow' else 1000}},indent=2))
 else:
  req=json.loads(Path(a.request).read_text());out=Path(req['output']);out.mkdir(parents=True,exist_ok=True);encoder=Encoder('vec768l12')
  predictor=utils.get_f0_predictor('rmvpe',hop_length=512,sampling_rate=44100,device='cuda',threshold=.05);results=[]
  for candidate in req['models']:
   net=model();net.load_state_dict(torch.load(candidate['checkpoint'],map_location='cpu',weights_only=False)['model'],strict=True);net.eval().requires_grad_(False)
   for variant in candidate['variants']:
    for case in req['cases']:
     folder=out/candidate['id']/variant['id']/case['id'];folder.mkdir(parents=True,exist_ok=True)
     x,sr=sf.read(case['source'],dtype='float32',always_2d=True);x=x.mean(axis=1);assert sr==44100
     y=np.zeros(len(x),np.float32);weights=np.zeros(len(x),np.float32)
     init=None
     if variant.get('initial_outputs'):
      init,_=sf.read(Path(variant['initial_outputs'])/case['id']/'converted.wav',dtype='float32')
     start=time.monotonic()
     with torch.inference_mode():
      for j,first in enumerate(range(0,len(x),int(19.5*sr))):
       last=min(len(x),first+20*sr);pad=22050;block=np.pad(x[max(0,first-pad):min(len(x),last+pad)],(max(0,pad-first),max(0,last+pad-len(x))))
       f0,uv=predictor.compute_f0_uv(block);n=len(f0);f=torch.tensor(f0,device='cuda',dtype=torch.float32)[None,:,None]
       u=encoder.encoder(torch.from_numpy(librosa.resample(block,orig_sr=sr,target_sr=16000)).cuda());u=utils.repeat_expand_2d(u.squeeze(0),n,'nearest').T[None]
       mel=None
       if init is not None:
        ib=np.pad(init[max(0,first-pad):min(len(init),last+pad)],(max(0,pad-first),max(0,last+pad-len(init))))
        wave=torch.from_numpy(ib).cuda()[None];mel=vocoder.extract(wave,sr)[:,:n]
        if variant.get('second_encoding'):
         u=encoder.encoder(torch.from_numpy(librosa.resample(ib,orig_sr=sr,target_sr=16000)).cuda());u=utils.repeat_expand_2d(u.squeeze(0),n,'nearest').T[None]
       else:wave=torch.from_numpy(block).cuda()[None]
       v=utils.Volume_Extractor(512).extract(wave)[None,:,None][:,:n].cuda()
       n=min(u.shape[1],f.shape[1],v.shape[1],mel.shape[1] if mel is not None else n)
       torch.manual_seed(42+j);m=net(u[:,:n],f[:,:n],v[:,:n],gt_spec=mel[:,:n] if mel is not None else None,infer=True,k_step=variant['k_step'],infer_speedup=10,method='dpm-solver',use_tqdm=False)
       b=vocoder.infer(m,f[:,:n])[0,0].float().cpu().numpy();b=np.pad(b,(0,max(0,len(block)-len(b))))[pad:pad+last-first]
       w=np.ones(last-first,np.float32);ov=min(22050,len(w))
       if first:w[:ov]*=np.linspace(1e-6,1,ov)
       if last<len(x):w[-ov:]*=np.linspace(1,1e-6,ov)
       y[first:last]+=b*w;weights[first:last]+=w
       if last==len(x):break
     y/=np.maximum(weights,1e-12);assert np.isfinite(y).all();peak=float(abs(y).max());sf.write(folder/'converted.wav',y*min(1.,.99/max(peak,1e-9)),sr,subtype='FLOAT')
     row={'model':candidate['id'],'variant':variant['id'],'case':case['id'],'source':case['source'],'output':str(folder/'converted.wav'),'role':case.get('role','conversion'),'status':'succeeded','wall_seconds':time.monotonic()-start}
     results.append(row);print(json.dumps(row),flush=True);(out/'outputs.json').write_text(json.dumps(results,indent=2))
   del net;torch.cuda.empty_cache()
