"""Bounded single-GPU training using upstream SVC models, dataset and losses."""
import json
import os
import random
import subprocess
import time
from pathlib import Path
from music_cover.common.paths import ROOT
from music_cover.common.io import write_json, identity
from music_cover.audio import sha256, decode
from music_cover.svc import checked_state
from core import runtime, build as build_model, Converter


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
        if config.get('experiment_warmstart'):
            checked_state(d,torch.load(config['experiment_base'],map_location='cpu',weights_only=False)['discriminator'])
        else:
            checked_state(d,torch.load(ROOT/'models/SoVitsSVC/D_0.pth',map_location='cpu',weights_only=True)['model'])
        kwargs=dict(lr=settings['learning_rate'],betas=tuple(config['train']['betas']),eps=config['train']['eps'])
        og=torch.optim.AdamW(g.parameters(),**kwargs);od=torch.optim.AdamW(d.parameters(),**kwargs)
        dataset=TextAudioSpeakerLoader(str(ROOT/c['dataset_dir']/'train.txt'),hps,all_in_mem=True,vol_aug=config['model'].get('vol_embedding',False))
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
            yh,ids,mask,(z,zp,mp,lp,mq,lq),pred_f0,norm_f0,lf0=g(content,f0,uv,spec_audio,g=speaker,c_lengths=lengths,spec_lengths=lengths,vol=volume.cuda() if config['model'].get('vol_embedding') else None)
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
