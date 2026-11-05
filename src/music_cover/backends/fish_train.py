"""Fish training uses upstream DiffSinger loss; data and optimizer state stay local."""
import csv
import html
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from ..audio import sha256
from ..common.paths import ROOT
from ..common.io import write_json, identity
from ..common.voice_paths import portable_path
from ..fish import MODEL_SPEC
from .fish_diffusion import runtime, FeatureExtractor, build_model, Converter


def path(value): return (ROOT/Path(value).expanduser()).resolve()


def extract_features(c,manifest,lock,out):
    spec={'dataset_identity':manifest['identity'],'model':c['model'],'augmentation':c['augmentation'],
          'seed':c['training']['seed'],'assets':{a['path']:a['sha256'] for a in lock['assets'] if a['scope']!='training'},
          'adapter_sha256':sha256(Path(__file__)),'runtime_sha256':sha256(Path(__file__).with_name('fish_diffusion.py'))}
    folder=path(c['dataset_dir'])/'features'
    folder.mkdir(exist_ok=True)
    tag=identity(spec)
    if (folder/'identity.json').exists() and json.loads((folder/'identity.json').read_text())!=spec:
        raise ValueError('Feature identity changed; use a new dataset directory')
    write_json(folder/'identity.json',spec)
    extractor=FeatureExtractor()
    rng=random.Random(c['training']['seed'])
    rows=[]
    for index,clip in enumerate(manifest['clips']):
        shifts=[0.0]
        probability=c['augmentation']['pitch_shift_probability'] if clip['split']=='train' else 0
        while probability>0:
            if rng.random()>probability:break
            shifts.append(rng.uniform(*c['augmentation']['key_shifts']));probability-=1
        wave,sr=sf.read(path(c['dataset_dir'])/clip['path'],dtype='float32')
        for variant,shift in enumerate(shifts):
            rel=f'features/{index:05d}_{variant}.npz';target=path(c['dataset_dir'])/rel
            meta=target.with_suffix('.json')
            expected={'identity':tag,'clip_sha256':clip['sha256'],'key_shift':shift}
            if target.exists() and meta.exists():
                saved=json.loads(meta.read_text())
                if saved['request']!=expected or sha256(target)!=saved['sha256']:
                    raise ValueError('Cached Fish features changed')
            else:
                data=extractor.extract(wave,shift)
                arrays={k:v.cpu().numpy() for k,v in data.items() if k!='key_shift'}
                if any(not np.isfinite(x).all() for x in arrays.values()):raise ValueError('Nonfinite Fish features')
                with target.with_suffix('.partial').open('wb') as stream:
                    np.savez_compressed(stream,**arrays,key_shift=np.float32(shift))
                target.with_suffix('.partial').replace(target)
                write_json(meta,{'request':expected,'sha256':sha256(target)})
            with np.load(target,allow_pickle=False) as a:
                n=a['mel'].shape[-1]
                if a['mel'].shape!=(128,n) or a['contents'].shape!=(256,n) or a['pitches'].shape!=(n,):
                    raise ValueError('Fish feature dimensions/frame counts differ')
                voiced=int(np.count_nonzero(a['pitches']>0))
            rows.append({'path':rel,'sha256':sha256(target),'frames':n,'voiced_frames':voiced,
                         'split':clip['split'],'group':clip['group'],'clip_path':clip['path'],'key_shift':shift})
        print(f'Fish features {index+1}/{len(manifest["clips"])} {clip["split"]}: {len(shifts)} variants',flush=True)
    write_json(out/'features.json',{**spec,'identity':tag,'features':rows,
        'runtime':{'python':__import__('sys').executable,'prefix':__import__('sys').prefix},
        'excluded_from_training':[r['path'] for r in rows if not r['voiced_frames']]})


def learning_rate(t,step):
    if step<t['warmup_steps']:
        return t['start_learning_rate']+(t['learning_rate']-t['start_learning_rate'])*step/t['warmup_steps']
    fraction=min(1.,(step-t['warmup_steps'])/(t['decay_steps']-t['warmup_steps']))
    return t['final_learning_rate']+.5*(t['learning_rate']-t['final_learning_rate'])*(1+math.cos(math.pi*fraction))


def load_features(c,report,split):
    result=[]
    for row in report['features']:
        if row['split']!=split or not row['voiced_frames']:continue
        with np.load(path(c['dataset_dir'])/row['path'],allow_pickle=False) as f:
            result.append({k:f[k].copy() for k in f.files})
    if not result:raise ValueError('No voiced features in '+split)
    return result


def batch(rows,crop,rng,torch):
    selected=[]
    for row in rows:
        length=min(crop,row['mel'].shape[-1])
        start=rng.randrange(max(1,row['mel'].shape[-1]-length+1))
        selected.append((row,start,length))
    lengths=[n for _,_,n in selected];width=max(lengths)
    def stack(key,channels):
        values=[]
        for row,start,length in selected:
            a=row[key][...,start:start+length]
            a=np.pad(a,((0,0),(0,width-length)) if channels else (0,width-length))
            values.append(a.T if channels else a[:,None])
        return torch.tensor(np.stack(values),device='cuda',dtype=torch.float32)
    lens=torch.tensor(lengths,device='cuda',dtype=torch.long)
    return dict(speakers=torch.zeros(len(rows),device='cuda',dtype=torch.long),
        contents=stack('contents',True),contents_lens=lens,contents_max_len=width,
        mel=stack('mel',True),mel_lens=lens,mel_max_len=width,pitches=stack('pitches',False),
        pitch_shift=torch.tensor([[float(r['key_shift'])] for r in rows],device='cuda'))


def train_model(c,run,out,features,resume=False):
    torch=runtime();t=c['training'];torch.manual_seed(t['seed']);random.seed(t['seed']);np.random.seed(t['seed'])
    rng=random.Random(t['seed'])
    checkpoints=run/'checkpoints';checkpoints.mkdir(exist_ok=True)
    existing=sorted(checkpoints.glob('step_*.ckpt'))
    if existing and not resume:raise ValueError('Training checkpoints exist; use --resume or a new run')
    payload=None
    if resume:
        if not existing:raise ValueError('No checkpoint to resume')
        payload=torch.load(existing[-1],map_location='cpu',weights_only=False)
        if payload['features_identity']!=features['identity']:raise ValueError('Resume feature identity mismatch')
        immutable=lambda value:{k:v for k,v in value.items() if k not in ('max_steps','save_every')}
        if immutable(payload['training'])!=immutable(t):raise ValueError('Resume optimizer/training parameters changed')
        model,coverage=build_model(existing[-1])
    else:
        model,coverage=build_model(ROOT/'models/FishDiffusion/content-vec-pretrained-v1.ckpt',pretrained=True)
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=t['start_learning_rate'],betas=(.9,.98),eps=1e-9,weight_decay=.01)
    step=0
    if payload:
        optimizer.load_state_dict(payload['optimizer']);step=payload['step']
        rng.setstate(payload['sampler_rng']);torch.set_rng_state(payload['torch_rng'])
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    if step>=t['max_steps']:raise ValueError('max_steps must exceed the resumed step')
    train=load_features(c,features,'train');valid=load_features(c,features,'validation')
    started=time.monotonic();losses=[]
    print(json.dumps({'python':__import__('sys').executable,'train_feature_count':len(train),
        'validation_feature_count':len(valid),'loading':coverage,'start_step':step,'max_steps':t['max_steps']}),flush=True)
    with (out/'losses.csv').open('w') as log:
        writer=csv.writer(log);writer.writerow(['step','train_loss','validation_loss','learning_rate','elapsed_s'])
        while step<t['max_steps']:
            step+=1
            lr=learning_rate(t,step)
            for group in optimizer.param_groups:group['lr']=lr
            b=batch([rng.choice(train) for _ in range(t['batch_size'])],t['crop_frames'],rng,torch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=t['precision']=='bfloat16'):
                loss=model(**b)['loss']
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite Fish training loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step();value=float(loss.detach());losses.append(value)
            validation_loss=None
            save=step%t['save_every']==0 or step==t['max_steps']
            if save:
                model.eval();values=[]
                with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(t['seed'])
                    for row in valid:
                        vb=batch([row],t['crop_frames'],random.Random(0),torch)
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=t['precision']=='bfloat16'):
                            values.append(float(model(**vb)['loss']))
                validation_loss=float(np.mean(values));model.train()
                if not math.isfinite(validation_loss):raise RuntimeError('Nonfinite validation loss')
                checkpoint=checkpoints/f'step_{step:08d}.ckpt'
                temporary=checkpoint.with_suffix('.partial')
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                    'training':t,'features_identity':features['identity'],'model_spec':MODEL_SPEC,
                    'sampler_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temporary)
                temporary.replace(checkpoint)
                write_json(out/'training.json',{'checkpoint':str(checkpoint.relative_to(run)),
                    'checkpoint_sha256':sha256(checkpoint),'step':step,'features_identity':features['identity'],
                    'loading':coverage,'trainable_parameters':sum(p.numel() for p in model.parameters()),
                    'frozen_components':['ContentVec','Parselmouth','NSF-HiFiGAN'],
                    'training':t,'validation_loss':validation_loss,'last_train_loss':value,
                    'runtime':{'python':__import__('sys').executable,'prefix':__import__('sys').prefix,'torch':torch.__version__},
                    'wall_seconds':time.monotonic()-started})
            if step==1 or step%25==0 or save:
                writer.writerow([step,value,validation_loss,lr,time.monotonic()-started]);log.flush()
                print(f'Fish step={step}/{t["max_steps"]} loss={value:.5f} valid={validation_loss} lr={lr:.7f} elapsed={time.monotonic()-started:.1f}s',flush=True)


def validate_model(c,run,out,features,checkpoint,split):
    torch=runtime();payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('features_identity')!=features['identity']:raise ValueError('Checkpoint belongs to another dataset')
    for row in c['validation']['source_files'].values():
        if sha256(path(row['path']))!=row['sha256']:raise ValueError('Validation source artifact changed')
    converter=Converter(checkpoint,{'model':MODEL_SPEC,'speaker_name':c['voice_id']})
    from .voice_metrics import VoiceMetrics
    try:metrics=VoiceMetrics();evaluation_error=None
    except Exception as error:metrics=None;evaluation_error=f'{type(error).__name__}: {error}'
    manifest=json.loads((path(c['dataset_dir'])/'manifest.json').read_text())
    held=[r for r in manifest['clips'] if r['split']==split]
    groups=list(dict.fromkeys(r['group'] for r in held))
    probes=[]
    # Each reconstruction probe is compared with other held-out recording groups.
    for group in groups[:2]:
        row=next(r for r in held if r['group']==group)
        refs=[path(c['dataset_dir'])/r['path'] for r in held if r['group']!=group][:3]
        if refs:probes.append((f'held_out_{len(probes)+1}',path(c['dataset_dir'])/row['path'],refs,'held_out_reconstruction'))
    source=path(c['validation']['source_files']['vocals']['path'])
    wave,sr=sf.read(source,dtype='float32',always_2d=True);wave=wave.mean(axis=1)
    references=[path(c['dataset_dir'])/r['path'] for r in held[:3]]
    for i,w in enumerate(c['validation']['windows'],1):
        p=out/f'song_window_{i:02d}_source.wav'
        sf.write(p,wave[round(w['start']*sr):round((w['start']+w['duration'])*sr)],sr,subtype='FLOAT')
        probes.append((f'song_window_{i:02d}',p,references,'source_song_conversion'))
    rows=[]
    for name,source,refs,kind in probes:
        folder=out/name;folder.mkdir()
        report=converter.infer(source,folder/'converted.wav',c['inference'],c['validation']['seed'])
        write_json(folder/'conversion.json',report)
        row={'id':name,'kind':kind,'audio':f'{name}/converted.wav','source':portable_path(source),
             'references':[portable_path(r) for r in refs],'generation_status':'succeeded','eligible':False}
        try:
            if evaluation_error:raise RuntimeError(evaluation_error)
            result=metrics.compare(source,folder/'converted.wav',refs,folder/'diagnostics')
            row.update(metrics=result,evaluation_status='succeeded',eligible=result['assessment']['status']=='passed')
        except Exception as error:
            row.update(evaluation_status='failed',evaluation_error=f'{type(error).__name__}: {error}')
        rows.append(row);print(json.dumps({'validation':name,'eligible':row['eligible'],
            'ecapa':row.get('metrics',{}).get('speaker_similarity'),'error':row.get('evaluation_error')}),flush=True)
    report={'schema_version':1,'backend':'fish_diffusion','checkpoint':portable_path(checkpoint),
        'checkpoint_sha256':sha256(checkpoint),'step':payload['step'],'features_identity':features['identity'],
        'evaluation_split':split,'selection_rule':'configured final step unless --checkpoint is explicit',
        'eligible':bool(rows) and all(r['eligible'] for r in rows),'blocks_delivery':False,
        'reference_independence':'held-out recording groups, not used as inference conditions',
        'subjective_quality':'awaiting_human_review','probes':rows}
    write_json(out/'validation.json',report)
    with (out/'metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['probe','kind','ecapa','pitch_status','eligible','evaluation_status'])
        for row in rows:
            m=row.get('metrics',{});writer.writerow([row['id'],row['kind'],m.get('speaker_similarity'),m.get('pitch',{}).get('status'),row['eligible'],row['evaluation_status']])
    body=''.join(f'<section><h2>{html.escape(r["id"])}</h2><audio controls src="{r["audio"]}"></audio><pre>{html.escape(json.dumps(r.get("metrics",r),ensure_ascii=False,indent=2))}</pre></section>' for r in rows)
    (out/'listening.html').write_text('<!doctype html><meta charset="utf-8"><title>Fish Diffusion validation</title><h1>Fish Diffusion validation</h1><p>Quality flags do not block audio export. Listening review is pending.</p>'+body)
