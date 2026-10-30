"""Bounded RVC training with upstream architecture, dataset and four losses."""
import json
import random
import time
from pathlib import Path
from ..common.paths import ROOT
from ..common.io import identity,write_json
from ..audio import sha256
from .rvc import runtime,Features,build_model,checked_load


def features(folder,manifest):
    import numpy as np
    import soundfile as sf
    import librosa
    with runtime():
        extractor=Features();lines=[];artifacts=[]
        for row in manifest['clips']:
            if row['split']!='train':continue
            wave,sr=sf.read(folder/row['path'],dtype='float32')
            audio16=librosa.resample(wave,orig_sr=sr,target_sr=16000)
            content,f0,coarse=extractor(audio16)
            stem=Path(row['path']).stem;paths=[]
            for name,value in [('content',content),('f0',f0),('coarse',coarse)]:
                p=folder/'features'/f'{stem}.{name}.npy';p.parent.mkdir(exist_ok=True)
                np.save(p,value,allow_pickle=False);paths.append(p)
                artifacts.append({'path':str(p.relative_to(folder)),'sha256':sha256(p)})
            a16=folder/'audio16'/f'{stem}.wav';a16.parent.mkdir(exist_ok=True);sf.write(a16,audio16,16000,subtype='FLOAT')
            artifacts.append({'path':str(a16.relative_to(folder)),'sha256':sha256(a16)})
            lines.append('|'.join([str(folder/row['path']),str(paths[0]),str(paths[2]),str(paths[1]),'0']))
            print('features: '+row['path'],flush=True)
    (folder/'train.txt').write_text('\n'.join(lines)+'\n')
    artifacts.append({'path':'train.txt','sha256':sha256(folder/'train.txt')})
    result={'dataset_identity':manifest['identity'],'artifacts':artifacts,'normalization':extractor.normalize,
            'feature_dim':768,'f0_policy':'unvoiced_zero','training_only':True}
    write_json(folder/'features.json',result);return result


def train(c,run,out,manifest,resume=False):
    import numpy as np
    import torch
    from torch.nn import functional as F
    settings=c['training'];config=json.loads((run/'model_config.json').read_text());folder=(ROOT/c['dataset_dir']).resolve()
    feature_manifest=json.loads((folder/'features.json').read_text())
    if feature_manifest['dataset_identity']!=manifest['identity']:raise ValueError('RVC feature dataset changed')
    for item in feature_manifest['artifacts']:
        if sha256(folder/item['path'])!=item['sha256']:raise ValueError('RVC feature changed')
    spec={'dataset_identity':manifest['identity'],'features_sha256':sha256(folder/'features.json'),'config':config,
          'training':{k:v for k,v in settings.items() if k not in ('max_steps','save_every')},
          'assets':json.loads((ROOT/'configs/rvc_assets.lock.json').read_text()),'trainer_sha256':sha256(__file__)}
    key=identity(spec);checkpoints=run/'checkpoints';checkpoints.mkdir(exist_ok=True);existing=sorted(checkpoints.glob('step_*.pth'))
    if existing and not resume:raise ValueError('RVC checkpoints exist; use --resume')
    if resume and not existing:raise ValueError('No RVC checkpoint to resume')
    torch.set_num_threads(4);seed=settings['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with runtime():
        from train.data_utils import TextAudioLoaderMultiNSFsid,TextAudioCollateMultiNSFsid
        from train.utils import HParams
        from train.losses import discriminator_loss,feature_loss,generator_loss,kl_loss
        from train.mel_processing import spec_to_mel_torch,mel_spectrogram_torch
        from infer.module.models import MultiPeriodDiscriminatorV2
        from infer.module import commons
        g,coverage=build_model(config,ROOT/'models/RVC/pretrained_v2/f0G48k.pth',pretrained=True)
        d=MultiPeriodDiscriminatorV2(config['model']['use_spectral_norm']).cuda()
        checked_load(d,torch.load(ROOT/'models/RVC/pretrained_v2/f0D48k.pth',map_location='cpu',weights_only=True)['model'])
        kw={'lr':settings['learning_rate'],'betas':tuple(config['train']['betas']),'eps':config['train']['eps']}
        og=torch.optim.AdamW(g.parameters(),**kw);od=torch.optim.AdamW(d.parameters(),**kw)
        dataset=TextAudioLoaderMultiNSFsid(str(folder/'train.txt'),HParams(**config['data']))
        cached=[dataset[i] for i in range(len(dataset))];collate=TextAudioCollateMultiNSFsid()
        print('Cached training clips: '+str(len(cached)),flush=True)
        before=g.enc_p.emb_phone.weight.detach().cpu().clone();step=0;order=[];cursor=0;resumed=None
        if resume:
            file=existing[-1];payload=torch.load(file,map_location='cpu',weights_only=False)
            if payload['training_identity']!=key:raise ValueError('RVC resume identity differs')
            checked_load(g,payload['model']);checked_load(d,payload['discriminator'])
            og.load_state_dict(payload['optimizer_g']);od.load_state_dict(payload['optimizer_d'])
            step=payload['step'];order=payload['order'];cursor=payload['cursor'];before=payload['initial_weight']
            random.setstate(payload['rng_python']);np.random.set_state(payload['rng_numpy']);torch.set_rng_state(payload['rng_torch']);torch.cuda.set_rng_state_all(payload['rng_cuda'])
            resumed={'path':str(file),'step':step,'sha256':sha256(file)};del payload
        initial=step
        if step>=settings['max_steps']:raise ValueError('max_steps must exceed restored step')
        g.train();d.train();last={}
        def save():
            file=checkpoints/f'step_{step:08d}.pth';temp=file.with_suffix('.tmp')
            payload={'backend':'rvc','model':g.state_dict(),'discriminator':d.state_dict(),'optimizer_g':og.state_dict(),'optimizer_d':od.state_dict(),
                     'step':step,'order':order,'cursor':cursor,'initial_weight':before,'training_identity':key,'dataset_identity':manifest['identity'],
                     'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
            torch.save(payload,temp);temp.replace(file)
            # Small strict inference checkpoint, independently checksum recorded.
            inference=checkpoints/f'model_{step:08d}.pth';torch.save({'model':{k:v.detach().cpu() for k,v in g.state_dict().items()},'step':step,'dataset_identity':manifest['identity']},inference)
            write_json(checkpoints/f'step_{step:08d}.json',{'step':step,'training_identity':key,'checkpoint_sha256':sha256(file),
                'inference':str(inference),'inference_sha256':sha256(inference),'losses':last})
            return file
        while step<settings['max_steps']:
            if cursor>=len(order):order=torch.randperm(len(cached)).tolist();cursor=0
            ids=order[cursor:cursor+settings['batch_size']];cursor+=len(ids);batch=[]
            for i in ids:
                spec_audio,wave,phone,pitch,pitchf,sid=cached[i];n=min(spec_audio.shape[-1],phone.shape[0]);width=min(n,settings['input_crop_frames'])
                first=random.randint(0,n-width)
                batch.append((spec_audio[:,first:first+width],wave[:,first*480:(first+width)*480],phone[first:first+width],pitch[first:first+width],pitchf[first:first+width],sid))
            phone,lengths,pitch,pitchf,spec_audio,spec_lengths,wave,_,sid=collate(batch)
            phone,lengths,pitch,pitchf,spec_audio,spec_lengths,wave,sid=[x.cuda() for x in (phone,lengths,pitch,pitchf,spec_audio,spec_lengths,wave,sid)]
            yh,offsets,mask,zmask,(z,zp,mp,lp,mq,lq)=g(phone,lengths,pitch,pitchf,spec_audio,spec_lengths,sid)
            mel=spec_to_mel_torch(spec_audio,2048,128,48000,0,None)
            ym=commons.slice_segments(mel,offsets,config['train']['segment_size']//480)
            yhm=mel_spectrogram_torch(yh[:,0],2048,128,48000,480,2048,0,None)
            target=commons.slice_segments(wave,offsets*480,config['train']['segment_size'])
            dr,dg,_,_=d(target,yh.detach());ld,_,_=discriminator_loss(dr,dg)
            if not torch.isfinite(ld):raise RuntimeError('Nonfinite RVC discriminator loss')
            od.zero_grad(set_to_none=True);ld.backward();torch.nn.utils.clip_grad_norm_(d.parameters(),1000,error_if_nonfinite=True);od.step()
            d.requires_grad_(False);dr,dg,fr,fg=d(target,yh)
            lm=F.l1_loss(ym,yhm)*config['train']['c_mel'];lk=kl_loss(zp,lq,mp,lp,zmask)*config['train']['c_kl'];lf=feature_loss(fr,fg);la,_=generator_loss(dg)
            lg=lm+lk+lf+la
            if not torch.isfinite(lg):raise RuntimeError('Nonfinite RVC generator loss')
            og.zero_grad(set_to_none=True);lg.backward();torch.nn.utils.clip_grad_norm_(g.parameters(),1000,error_if_nonfinite=True);og.step();d.requires_grad_(True)
            step+=1;lr=settings['learning_rate']*config['train']['lr_decay']**(step*settings['batch_size']/len(cached))
            for optimizer in (og,od):
                for group in optimizer.param_groups:group['lr']=lr
            if step%10==0 or step==initial+1 or step==settings['max_steps']:
                last={'step':step,'generator':float(lg.detach()),'discriminator':float(ld.detach()),'mel':float(lm.detach()),'kl':float(lk.detach()),'feature':float(lf.detach()),'adversarial':float(la.detach()),'lr':lr,'elapsed_s':time.monotonic()-start}
                print(json.dumps(last),flush=True)
                with (out/'losses.jsonl').open('a') as stream:stream.write(json.dumps(last)+'\n')
            if step%settings['save_every']==0 or step==settings['max_steps']:last_checkpoint=save()
        delta=float(torch.linalg.vector_norm(g.enc_p.emb_phone.weight.detach().cpu()-before))
        if not delta>0:raise RuntimeError('RVC parameters did not update')
        report={'step':step,'initial_step':initial,'training_identity':key,'dataset_identity':manifest['identity'],'resumed_from':resumed,
                'checkpoint':str(last_checkpoint),'core_weight_delta_l2':delta,'parameter_loading':coverage,'wall_seconds':time.monotonic()-start,
                'precision':'FP32 with TF32','losses':last,'generator_parameters':sum(p.numel() for p in g.parameters()),
                'discriminator_parameters':sum(p.numel() for p in d.parameters()),'frozen':['HuBERT/ContentVec','RMVPE'],
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
        write_json(out/'training.json',report)
