import json
from pathlib import Path
import torch,numpy as np,soundfile as sf
from core import ROOT,EXP,runtime,Converter
from music_cover.backends.so_vits_svc import Converter as Production
p=ROOT/'models/voices/yang_kun/so_vits_svc/optimized_v002_20261028';selection=json.loads((EXP/'selection.json').read_text());m=selection['model_request'];cfg=json.loads((p/'config.json').read_text())
a=torch.load(p/'model.pth',map_location='cpu',weights_only=False)['model'];b=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)['model'];assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a);del a,b
out=EXP/'parity_probe';out.mkdir(exist_ok=True)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
with runtime(out/'runtime'):
 c=Converter(m['checkpoint'],cfg);c.infer(EXP/'cases/hero_celtic/source.wav',out/'experiment.wav',selection['variant']['settings']);del c;torch.cuda.empty_cache()
 c=Production(p/'model.pth',cfg,'yang_kun');c.infer(EXP/'cases/hero_celtic/source.wav',out/'production.wav',{'noise_scale':.1,'chunk_seconds':8.,'overlap_seconds':.2},42)
a,sr=sf.read(out/'experiment.wav',dtype='float32');b,_=sf.read(out/'production.wav',dtype='float32');error=float(np.max(abs(a-b)));rms=float(np.sqrt(np.mean((a-b)**2)))
result={'all_exported_model_tensors_equal':True,'same_process_deterministic_cudnn':True,'max_abs_error':error,'rms_error':rms,'snr_db':float(20*np.log10(max(np.sqrt(np.mean(a*a)),1e-12)/max(rms,1e-12)))}
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert error<1e-5
