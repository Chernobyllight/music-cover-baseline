#!/usr/bin/env python3
"""Check ContentVec's Transformers port against independent original-weight equations."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json,pickle,types,pathlib,torch
from torch.nn import functional as F
from music_cover.backends.svc_contentvec import ContentVec,_ContentVecUnpickler
from music_cover.common.paths import ROOT
from music_cover.common.io import write_json
torch.set_num_threads(4)
e=ContentVec(device='cpu');mod=types.ModuleType('contentvec_pickle');mod.Unpickler=_ContentVecUnpickler;mod.load=pickle.load
s={k:v.float() for k,v in torch.load(ROOT/'models/SoVitsSVC/contentvec.pt',map_location='cpu',pickle_module=mod,weights_only=False)['model'].items()}
def linear(x,p):return F.linear(x,s[p+'.weight'],s[p+'.bias'])
def norm(x,p):return F.layer_norm(x,(x.shape[-1],),s[p+'.weight'],s[p+'.bias'],1e-5)
def reference(wave, layer=12):
 x=wave[None,None]
 for i,stride in enumerate([5,2,2,2,2,2,2]):
  p=f'feature_extractor.conv_layers.{i}';x=F.conv1d(x,s[p+'.0.weight'],stride=stride)
  if i==0:x=F.group_norm(x,512,s[p+'.2.weight'],s[p+'.2.bias'],1e-5)
  x=F.gelu(x)
 x=linear(norm(x.transpose(1,2),'layer_norm'),'post_extract_proj')
 w=torch._weight_norm(s['encoder.pos_conv.0.weight_v'],s['encoder.pos_conv.0.weight_g'],2)
 pos=F.gelu(F.conv1d(x.transpose(1,2),w,s['encoder.pos_conv.0.bias'],padding=64,groups=16)[:,:,:-1]).transpose(1,2)
 x=norm(x+pos,'encoder.layer_norm')
 for i in range(layer):
  p=f'encoder.layers.{i}';b,t,d=x.shape
  q,k,v=[linear(x,p+'.self_attn.'+a+'_proj').view(b,t,12,64).transpose(1,2) for a in ['q','k','v']]
  attn=torch.softmax((q*(64**-.5))@k.transpose(-2,-1),dim=-1)
  y=(attn@v).transpose(1,2).reshape(b,t,d)
  x=norm(x+linear(y,p+'.self_attn.out_proj'),p+'.self_attn_layer_norm')
  x=norm(x+linear(F.gelu(linear(x,p+'.fc1')),p+'.fc2'),p+'.final_layer_norm')
 if layer==9:x=linear(x,'final_proj')
 return x.transpose(1,2)
import soundfile as sf,librosa
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--audio',type=Path)
parser.add_argument('--output',type=Path,default=ROOT/'cache/svc_setup/contentvec_equivalence.json')
args=parser.parse_args()
cases=[torch.randn(16000,generator=torch.Generator().manual_seed(42))*.01]
if args.audio:
 wave,sr=sf.read(args.audio,dtype='float32',always_2d=True)
 cases.append(torch.from_numpy(librosa.resample(wave[:sr*2].mean(axis=1),orig_sr=sr,target_sr=16000)))
rows=[]
with torch.inference_mode():
 for variant,layer in [('vec768l12',12),('vec256l9',9)]:
  e=ContentVec(device='cpu',variant=variant)
  for w in cases:
   a=e.encoder(w);b=reference(w,layer);error=float((a-b).abs().max());rms=float((a-b).square().mean().sqrt())
   rows.append({'variant':variant,'samples':len(w),'shape':list(a.shape),'maximum_absolute_difference':error,'rms_difference':rms})
   assert torch.allclose(a,b,atol=2e-4,rtol=2e-4),(error,rms)
write_json(args.output,{'method':'independent functional evaluation of original fairseq ContentVec inference equations vs Transformers port; CPU FP32','cases':rows,'passed':True})
print(json.dumps(rows))
