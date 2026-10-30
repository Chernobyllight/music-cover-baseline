"""Bounded single-GPU training using upstream SVC models, dataset and losses."""
import json
import os
import random
import subprocess
import time
from pathlib import Path
from ..common.paths import ROOT
from ..common.io import write_json, identity
from ..audio import sha256, decode
from ..svc import checked_state
from .so_vits_svc import runtime, build_model, Converter


def prepare_features(folder, clips, workspace):
    import torch
    import soundfile as sf
    import librosa
    import numpy as np
    from .svc_contentvec import ContentVec
    with runtime(workspace):
        import utils
        from modules.mel_processing import spectrogram_torch
        encoder=ContentVec();pitch=utils.get_f0_predictor('rmvpe',hop_length=512,sampling_rate=44100,device='cuda',threshold=.05)
        for row in clips:
            if row['split']!='train':continue
            file=folder/row['path'];audio,sr=sf.read(file,dtype='float32')
            with torch.inference_mode():
                wave=torch.from_numpy(audio).to('cuda')
                content=encoder.encoder(torch.from_numpy(librosa.resample(audio,orig_sr=sr,target_sr=16000)).to('cuda'))
                f0,uv=pitch.compute_f0_uv(audio)
                spec=spectrogram_torch(wave[None],2048,44100,512,2048,center=False)[0]
            torch.save(content.cpu(),str(file)+'.soft.pt')
            torch.save(spec.cpu(),file.with_suffix('.spec.pt'))
            np.save(str(file)+'.f0.npy',np.stack([f0,uv]))
            print('features: '+row['path'],flush=True)


def train_model(c,run,out,manifest,resume=False):
    import numpy as np
    import torch
    from torch.nn import functional as F
    settings=c['training'];config=json.loads((run/'model_config.json').read_text())
    lock=json.loads((ROOT/'configs/so_vits_svc_assets.lock.json').read_text())
    spec={'dataset_identity':manifest['identity'],'config':config,'training':{k:v for k,v in settings.items() if k not in ('max_steps','save_every')},
          'base_assets':lock,'trainer_sha256':sha256(Path(__file__))}
    key=identity(spec);checkpoint_dir=run/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
    existing=sorted(checkpoint_dir.glob('step_*.pth'))
    if existing and not resume:raise ValueError('Checkpoints exist; explicitly pass --resume')
    if resume and not existing:raise ValueError('No checkpoint to resume')
    torch.set_num_threads(4);seed=int(settings.get('seed',42))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with runtime(run/'train_runtime'):
        from data_utils import TextAudioSpeakerLoader,TextAudioCollate
        from models import MultiPeriodDiscriminator
        from modules import commons
        from modules.losses import discriminator_loss,feature_loss,generator_loss,kl_loss
        from modules.mel_processing import spec_to_mel_torch,mel_spectrogram_torch
        import utils
        hps=utils.HParams(**config)
        g,coverage=build_model(config,ROOT/'models/SoVitsSVC/G_0.pth',replace_speaker=True)
        d=MultiPeriodDiscriminator(config['model']['use_spectral_norm']).cuda()
        checked_state(d,torch.load(ROOT/'models/SoVitsSVC/D_0.pth',map_location='cpu',weights_only=True)['model'])
        kwargs=dict(lr=settings['learning_rate'],betas=tuple(config['train']['betas']),eps=config['train']['eps'])
        og=torch.optim.AdamW(g.parameters(),**kwargs);od=torch.optim.AdamW(d.parameters(),**kwargs)
        dataset=TextAudioSpeakerLoader(str(Path(c['dataset_dir']) if False else ROOT/c['dataset_dir']/'train.txt'),hps,all_in_mem=True,vol_aug=False)
        collate=TextAudioCollate();step=0;order=[];cursor=0;resumed_from=None
        before=g.pre.weight.detach().cpu().clone()
        if resume:
            f=existing[-1];payload=torch.load(f,map_location='cpu',weights_only=False)
            if payload['training_identity']!=key:raise ValueError('Resume configuration/dataset/weights/code differs')
            checked_state(g,payload['model']);checked_state(d,payload['discriminator'])
            og.load_state_dict(payload['optimizer_g']);od.load_state_dict(payload['optimizer_d'])
            step=payload['step'];order=payload['order'];cursor=payload['cursor'];before=payload['initial_pre_weight']
            random.setstate(payload['rng_python']);np.random.set_state(payload['rng_numpy']);torch.set_rng_state(payload['rng_torch'])
            torch.cuda.set_rng_state_all(payload['rng_cuda']);resumed_from={'checkpoint':str(f),'step':step,'sha256':sha256(f)}
            del payload
        initial_step=step
        if settings['max_steps']<=step:raise ValueError('max_steps must exceed restored step')
        g.train();d.train();history=[];last_checkpoint=None
        def save():
            destination=checkpoint_dir/f'step_{step:08d}.pth';tmp=destination.with_suffix('.tmp')
            payload={'model':g.state_dict(),'discriminator':d.state_dict(),'optimizer_g':og.state_dict(),'optimizer_d':od.state_dict(),
                'step':step,'order':order,'cursor':cursor,'training_identity':key,'dataset_identity':manifest['identity'],
                'initial_pre_weight':before,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
                'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
                'backend':'so_vits_svc','adaptation_scope':'speaker'}
            torch.save(payload,tmp);tmp.replace(destination);return destination
        while step<settings['max_steps']:
            if cursor>=len(order):order=torch.randperm(len(dataset)).tolist();cursor=0
            indexes=order[cursor:cursor+settings['batch_size']];cursor+=len(indexes)
            batch=collate([dataset[i] for i in indexes])
            content,f0,spec_audio,y,speaker,lengths,uv,volume=batch
            content,f0,spec_audio,y,speaker,lengths,uv=[x.cuda() for x in (content,f0,spec_audio,y,speaker,lengths,uv)]
            yh,ids,mask,(z,zp,mp,lp,mq,lq),pred_f0,norm_f0,lf0=g(content,f0,uv,spec_audio,g=speaker,c_lengths=lengths,spec_lengths=lengths)
            mel=spec_to_mel_torch(spec_audio,2048,80,44100,0,22050)
            target_mel=commons.slice_segments(mel,ids,config['train']['segment_size']//512)
            generated_mel=mel_spectrogram_torch(yh.squeeze(1),2048,80,44100,512,2048,0,22050)
            target=commons.slice_segments(y,ids*512,config['train']['segment_size'])
            dr,dg,_,_=d(target,yh.detach());ld,_,_=discriminator_loss(dr,dg)
            if not torch.isfinite(ld):raise RuntimeError('Nonfinite discriminator loss')
            od.zero_grad(set_to_none=True);ld.backward();torch.nn.utils.clip_grad_norm_(d.parameters(),1000,error_if_nonfinite=True);od.step()
            for parameter in d.parameters():parameter.requires_grad_(False)
            dr,dg,fr,fg=d(target,yh)
            lm=F.l1_loss(target_mel,generated_mel)*config['train']['c_mel']
            lk=kl_loss(zp,lq,mp,lp,mask)*config['train']['c_kl'];lf=feature_loss(fr,fg);la,_=generator_loss(dg)
            lfp=F.mse_loss(pred_f0,lf0);lg=lm+lk+lf+la+lfp
            if not torch.isfinite(lg):raise RuntimeError('Nonfinite generator loss')
            og.zero_grad(set_to_none=True);lg.backward();torch.nn.utils.clip_grad_norm_(g.parameters(),1000,error_if_nonfinite=True);og.step()
            for parameter in d.parameters():parameter.requires_grad_(True)
            step+=1
            # Match the upstream epoch-wise exponential schedule with the effective dataset epoch.
            lr=settings['learning_rate']*config['train']['lr_decay']**(step*settings['batch_size']/len(dataset))
            for optimizer in (og,od):
                for group in optimizer.param_groups:group['lr']=lr
            if step%10==0 or step==initial_step+1 or step==settings['max_steps']:
                row={'step':step,'generator':float(lg.detach()),'discriminator':float(ld.detach()),'mel':float(lm.detach()),'lr':lr}
                history.append(row);print(json.dumps(row),flush=True)
                with (out/'losses.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if step%settings['save_every']==0 or step==settings['max_steps']:last_checkpoint=save()
        delta=float(torch.linalg.vector_norm(g.pre.weight.detach().cpu()-before))
        if delta==0:raise RuntimeError('SVC core parameters did not update')
        write_json(out/'training.json',{'training_identity':key,'dataset_identity':manifest['identity'],'step':step,
            'initial_step':initial_step,'resumed_from':resumed_from,'checkpoint':str(last_checkpoint),'checkpoint_sha256':sha256(last_checkpoint),
            'parameter_loading':coverage,'core_pre_weight_delta_l2':delta,'wall_seconds':time.monotonic()-start,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'training_scope':'all generator and discriminator parameters; fixed ContentVec and RMVPE',
            'last_losses':history[-1]})


def validate_model(c,run,out,manifest,requested,split):
    import torch
    from ..svc_training import path
    candidates=[path(x) for x in requested] if requested else [sorted((run/'checkpoints').glob('step_*.pth'))[-1]]
    state=json.loads((run/'state.json').read_text());trained=json.loads((Path(state['stages']['01_train']['output'])/'training.json').read_text())
    if split=='test':
        reports=[json.loads(f.read_text()) for f in (run/'stages/02_validate').glob('attempt_*/artifacts/validation.json')]
        if len(candidates)!=1 or not any(r['evaluation_split']=='validation' and r['selected_checkpoint']==str(candidates[0]) and r['dataset_identity']==manifest['identity'] for r in reports):
            raise ValueError('Test requires one checkpoint previously selected on validation')
    refs=[str(path(c['dataset_dir'])/r['path']) for r in manifest['clips'] if r['split']==split]
    source_run=path(c['validation']['source_run']);state=json.loads((source_run/'stage_state.json').read_text())
    item=state['stages']['04_vocal_separation']['outputs']['vocals']
    if sha256(item['path'])!=item['sha256']:raise ValueError('Source stage audio changed')
    cases=[]
    for i,window in enumerate(c['validation']['windows'],1):
        source=out/f'source_{i:03d}.wav';decode(item['path'],source,rate=44100,channels=1,start=window['start'],duration=window['duration'])
        cases.append(source)
    config=json.loads((run/'model_config.json').read_text());rows=[];jobs=[]
    with runtime(run/'validation_runtime'):
        for checkpoint in candidates:
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if payload['training_identity']!=trained['training_identity']:raise ValueError('Candidate from different training run')
            del payload
            converter=Converter(checkpoint,config,c['voice_id'])
            row={'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'cases':[]};rows.append(row)
            for i,source in enumerate(cases,1):
                folder=out/checkpoint.stem/f'case_{i:03d}';folder.mkdir(parents=True)
                conversion=converter.infer(source,folder/'converted.wav',seed=42)
                write_json(folder/'conversion.json',conversion)
                jobs.append({'source':str(source),'converted':str(folder/'converted.wav'),'references':refs,'output':str(folder/'diagnostics'),
                             'checkpoint':str(checkpoint),'case':i})
            del converter;torch.cuda.empty_cache()
    request=out/'metrics_request.json';write_json(request,{'jobs':jobs})
    command=[str(ROOT/'envs/seed-vc/bin/python'),'-m','music_cover.backends.cli','svc_metrics','--request',str(request),'--output',str(out)]
    with (out/'metrics.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
    scored=json.loads((out/'metrics.json').read_text())
    for row in rows:
        row['cases']=[x for x in scored['cases'] if x['checkpoint']==row['checkpoint']]
        row['eligible']=bool(row['cases']) and all(x['eligible'] for x in row['cases'])
        row['mean_ecapa']=sum(x['metrics']['speaker_similarity'] for x in row['cases'])/len(row['cases'])
    eligible=[r for r in rows if r['eligible']];selected=max(eligible,key=lambda r:r['mean_ecapa']) if eligible else None
    write_json(out/'validation.json',{'backend':'so_vits_svc','dataset_identity':manifest['identity'],'evaluation_split':split,
        'source_run':str(source_run),'source_vocals_sha256':item['sha256'],'reference_files':refs,'candidates':rows,
        'selected_checkpoint':selected['checkpoint'] if selected else None,'selection':'highest mean ECAPA among technically eligible checkpoints',
        'subjective_quality':'awaiting_human_review','reference_used_for_inference':False})
    print('validation selected: '+str(selected['checkpoint'] if selected else None),flush=True)
