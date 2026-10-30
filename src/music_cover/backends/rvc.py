"""Headless RVC v2: shared train/inference features, explicit UV, exact retrieval."""
import contextlib
import json
import os
import sys
import time
import types
from pathlib import Path
from ..common.paths import ROOT
from ..common.io import write_json
from ..audio import sha256, inspect_audio
from ..rvc import check_config, check_assets, validate_settings, UPSTREAM_COMMIT


@contextlib.contextmanager
def runtime():
    upstream=str(ROOT/'third_party/rvc'); old=os.environ.get('RVC_CUDA_GRAPH')
    os.environ['RVC_CUDA_GRAPH']='0'; sys.path.insert(0,upstream)
    prefixes=('infer','tools','train','i18n')
    saved={k:v for k,v in sys.modules.items() if any(k==p or k.startswith(p+'.') for p in prefixes)}
    for key in saved: del sys.modules[key]
    # Upstream uses namespace directories. An unrelated installed `tools`
    # regular package otherwise wins despite sys.path precedence.
    for name in prefixes:
        module=types.ModuleType(name);module.__path__=[str(Path(upstream)/name)];sys.modules[name]=module
    try: yield
    finally:
        for key in list(sys.modules):
            if any(key==p or key.startswith(p+'.') for p in prefixes):del sys.modules[key]
        sys.modules.update(saved)
        sys.path.remove(upstream)
        if old is None: os.environ.pop('RVC_CUDA_GRAPH',None)
        else: os.environ['RVC_CUDA_GRAPH']=old


def checked_load(model, saved, replace_speaker=False):
    current=model.state_dict(); saved=dict(saved)
    if set(current)!=set(saved): raise ValueError('RVC checkpoint keys differ')
    if replace_speaker and saved['emb_g.weight'].shape!=current['emb_g.weight'].shape:
        if saved['emb_g.weight'].shape[1:]!=current['emb_g.weight'].shape[1:]: raise ValueError('RVC speaker embedding shape mismatch')
        saved['emb_g.weight']=saved['emb_g.weight'][:1].clone()
    bad=[k for k in current if current[k].shape!=saved[k].shape]
    if bad: raise ValueError('RVC checkpoint shape mismatch: '+str(bad[:5]))
    model.load_state_dict(saved,strict=True)
    return {'matched_tensors':len(current),'speaker_embedding_first_row':replace_speaker,'strict':True}


def build_model(config, checkpoint=None, pretrained=False):
    import torch
    from infer.module.models import SynthesizerTrnMs768NSFsid
    check_config(config)
    g=SynthesizerTrnMs768NSFsid(config['data']['filter_length']//2+1,
        config['train']['segment_size']//480,**config['model'],is_half=False,sr=48000)
    coverage=None
    if checkpoint:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        coverage=checked_load(g,payload['model'],replace_speaker=pretrained)
    return g.cuda(),coverage


def pitch_to_coarse(f0):
    import numpy as np
    mel=1127*np.log1p(f0/700); lo=1127*np.log1p(50/700); hi=1127*np.log1p(1100/700)
    mel[mel>0]=(mel[mel>0]-lo)*254/(hi-lo)+1
    return np.clip(np.rint(mel),1,255).astype('int64')


class Features:
    def __init__(self):
        import torch
        from infer import hubert
        from infer.rmvpe import RMVPE
        hubert.HUBERT_MODEL_PATH=ROOT/'models/RVC/hubert_base'
        hubert.hubert_audio_requires_normalization.cache_clear()
        self.model=hubert.load_hubert_model('cuda',False).requires_grad_(False)
        self.normalize=hubert.hubert_audio_requires_normalization()
        self.extract=hubert.extract_hubert_features
        self.pitch=RMVPE(str(ROOT/'models/RVC/rmvpe.pt'),is_half=False,device='cuda')

    def __call__(self,audio16):
        import numpy as np
        import torch
        from torch.nn import functional as F
        x=torch.from_numpy(np.asarray(audio16,dtype='float32')).cuda()[None]
        if self.normalize: x=F.layer_norm(x,x.shape[1:])
        with torch.inference_mode(): features=self.extract(self.model,x,'v2').squeeze(0).float().cpu().numpy()
        f0=self.pitch.infer_from_audio(np.asarray(audio16,dtype='float32'),thred=.03)
        f0=np.asarray(f0,dtype='float32')
        # Keep RMVPE's zero-valued unvoiced frames in BOTH training and inference.
        # The mask must not be reconstructed from an interpolated contour.
        if not np.isfinite(features).all() or not np.isfinite(f0).all(): raise ValueError('Nonfinite RVC features')
        return features,f0,pitch_to_coarse(f0)


def retrieve(features,index,vectors,rate):
    import numpy as np
    if rate==0: return features.copy()
    if index is None or vectors is None or index.d!=768 or index.ntotal<8: raise ValueError('Invalid RVC retrieval index')
    distances,ids=index.search(np.ascontiguousarray(features,dtype='float32'),8)
    if (ids<0).any() or not np.isfinite(distances).all(): raise ValueError('Invalid RVC nearest neighbors')
    weights=1/np.maximum(distances.astype('float64'),1e-8)**2
    weights/=weights.sum(axis=1,keepdims=True)
    nearest=(vectors[ids]*weights[...,None]).sum(axis=1).astype('float32')
    return features*(1-rate)+nearest*rate


class Converter:
    def __init__(self,checkpoint,config,index=None):
        import faiss
        self.config=config;self.model,self.coverage=build_model(config,checkpoint)
        self.model.eval().requires_grad_(False);self.features=Features();self.index=None;self.vectors=None
        if index:
            faiss.omp_set_num_threads(4);self.index=faiss.read_index(str(index))
            if self.index.d!=768 or self.index.ntotal<8:raise ValueError('RVC index dimension/count invalid')
            self.vectors=self.index.reconstruct_n(0,self.index.ntotal)

    def infer(self,source,output,settings=None,seed=42):
        import numpy as np
        import librosa
        import soundfile as sf
        import torch
        p=validate_settings(settings or {}); torch.set_num_threads(4)
        wave,sr=sf.read(source,dtype='float32',always_2d=True)
        if sr!=44100 or not len(wave) or not np.isfinite(wave).all():raise ValueError('RVC expects finite 44.1 kHz input')
        if p['index_rate'] and self.index is None:raise ValueError('RVC retrieval requested without index')
        wave=wave.mean(1); native=librosa.resample(wave,orig_sr=sr,target_sr=48000)
        width=round(p['chunk_seconds']*48000);overlap=round(p['overlap_seconds']*48000);pad=round(p['pad_seconds']*48000)
        result=np.zeros(len(native),np.float32);weights=np.zeros(len(native),np.float32);segments=[];start=time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for number,first in enumerate(range(0,len(native),width-overlap)):
                last=min(len(native),first+width);core=native[first:last]
                if np.sqrt(np.mean(core**2))<1e-5: audio=np.zeros_like(core);status='silent';uv_count=0
                else:
                    block=np.pad(native[max(0,first-pad):min(len(native),last+pad)],(max(0,pad-first),max(0,last+pad-len(native))))
                    feature,f0,coarse=self.features(librosa.resample(block,orig_sr=48000,target_sr=16000))
                    mixed=retrieve(feature,self.index,self.vectors,p['index_rate'])
                    # Same 50 Hz -> 100 Hz repeat expansion as upstream training.
                    raw=np.repeat(feature,2,axis=0);mixed=np.repeat(mixed,2,axis=0)
                    n=min(len(mixed),len(f0),len(block)//480);mixed=mixed[:n];raw=raw[:n];f0=f0[:n];coarse=coarse[:n]
                    uv=f0==0;uv_count=int(uv.sum())
                    if p['protect']<.5: mixed[uv]=mixed[uv]*p['protect']+raw[uv]*(1-p['protect'])
                    torch.manual_seed((seed+number)%2**32);torch.cuda.manual_seed_all((seed+number)%2**32)
                    t=lambda x:torch.from_numpy(np.ascontiguousarray(x)).cuda()[None]
                    generated=self.model.infer(t(mixed),torch.tensor([n],device='cuda'),t(coarse),t(f0),torch.tensor([0],device='cuda'))[0]
                    audio=generated[0,0].float().cpu().numpy()
                    if abs(len(audio)-len(block))>1920:raise RuntimeError('RVC frame count mismatch')
                    audio=np.pad(audio,(0,max(0,len(block)-len(audio))))[pad:pad+len(core)];status='converted'
                weight=np.ones(len(core),np.float32);nfade=min(overlap,len(core))
                if first:weight[:nfade]*=np.linspace(1e-6,1,nfade)
                if last<len(native):weight[-nfade:]*=np.linspace(1,1e-6,nfade)
                result[first:last]+=audio*weight;weights[first:last]+=weight
                segments.append({'first_sample':first,'last_sample':last,'sample_rate':48000,'status':status,'unvoiced_frames':uv_count})
                if last==len(native):break
        result/=np.maximum(weights,1e-12)
        if not np.isfinite(result).all():raise RuntimeError('Nonfinite RVC output')
        peak=float(np.max(np.abs(result)));gain=min(1.,.99/max(peak,1e-12));result*=gain
        output=Path(output);sf.write(output.with_name('converted_native_48k.wav'),result,48000,subtype='FLOAT')
        final=librosa.resample(result,orig_sr=48000,target_sr=44100)
        final=np.pad(final,(0,max(0,len(wave)-len(final))))[:len(wave)]
        sf.write(output,final,44100,subtype='FLOAT')
        return {'settings':p,'segments':segments,'duration_s':len(final)/44100,'wall_seconds':time.monotonic()-start,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'pre_limit_peak':peak,'peak_gain':gain,
                'parameter_loading':self.coverage,'feature_normalization':self.features.normalize,'f0_policy':'unvoiced_zero',
                'retrieval':{'enabled':p['index_rate']>0,'k':8,'vectors':self.index.ntotal if self.index else 0},'native_sample_rate':48000}


def convert(request,output):
    from ..voice import resolve_voice,voice_record
    check_assets();voice=resolve_voice('finetuned',request['voice_profile'],backend='rvc');output=Path(output)
    with runtime():
        converter=Converter(voice['checkpoint'],json.loads(voice['config'].read_text()),voice['index'])
        report=converter.infer(request['source'],output/'converted.wav',request.get('rvc'),request['seed'])
    write_json(output/'conversion.json',{**voice_record(voice),**report,'upstream_commit':UPSTREAM_COMMIT,
        'source_sha256':sha256(request['source']),'reference_sha256':sha256(request['reference']),
        'seed':request['seed'],'semitones':0,'auto_f0_adjust':False,'training_performed':False,
        'reference_used_for_inference':False,'reference_role':'evaluation_and_pitch_adaptation',
        'diffusion_steps':None,'audio':inspect_audio(output/'converted.wav')})
