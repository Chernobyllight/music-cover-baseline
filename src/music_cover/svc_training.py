"""Explicit SVC prepare/train/validate/export, separate from Seed-VC adaptation."""
import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
import yaml
from .common.paths import ROOT
from .common.io import write_json, identity
from .audio import sha256, decode, inspect_audio
from .svc import check_assets, UPSTREAM_COMMIT, check_model_config


def path(value): return (ROOT/Path(value).expanduser()).resolve()


def load_config(filename):
    c = yaml.safe_load(path(filename).read_text())
    for key in ('voice_id', 'version', 'source_dataset', 'dataset_dir', 'reference', 'training', 'validation'):
        if key not in c: raise ValueError('Missing SVC config: '+key)
    for key in ('voice_id','version'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', c[key]): raise ValueError('Unsafe '+key)
    t=c['training']
    for key in ('max_steps','save_every','batch_size'):
        if type(t[key]) is not int or t[key] < 1: raise ValueError('Invalid '+key)
    t['learning_rate']=float(t.get('learning_rate',1e-4))
    if not 0 < t['learning_rate'] <= 1e-3: raise ValueError('Invalid learning_rate')
    return c


def source_manifest(c):
    folder=path(c['source_dataset']); m=json.loads((folder/'manifest.json').read_text())
    if m['voice_id'] != c['voice_id'] or m.get('smoke_test'): raise ValueError('Require a matching held-out singer dataset')
    groups={}; hashes={}
    for row in m['clips']:
        f=(folder/row['path']).resolve()
        if not f.is_relative_to(folder) or sha256(f)!=row['sha256']: raise ValueError('Source dataset changed or invalid path')
        for table,key in ((groups,row['group']),(hashes,row['sha256'])):
            if key in table and table[key]!=row['split']: raise ValueError('Dataset split leakage')
            table[key]=row['split']
    if not {'train','validation','test'} <= set(groups.values()): raise ValueError('Need train/validation/test groups')
    if sha256(path(c['reference'])) != m['reference_sha256']: raise ValueError('Reference differs from source manifest')
    return m


def prepared(c):
    m=json.loads((path(c['dataset_dir'])/'manifest.json').read_text())
    original=source_manifest(c)
    if m['source_manifest_sha256']!=sha256(path(c['source_dataset'])/'manifest.json'): raise ValueError('Prepared source manifest changed')
    for row in m['artifacts']:
        f=path(c['dataset_dir'])/row['path']
        if not f.is_file() or sha256(f)!=row['sha256']: raise ValueError('Prepared SVC artifact changed: '+str(f))
    return m


def prepare(c,run,out,args):
    from .backends.svc_train import prepare_features
    lock=check_assets(training=True); original=source_manifest(c)
    destination=path(c['dataset_dir'])
    spec={'source_manifest_sha256':sha256(path(c['source_dataset'])/'manifest.json'),
          'voice_id':c['voice_id'],'assets':lock,'format':'44100-mono-PCM16',
          'feature_adapter_sha256':sha256(ROOT/'src/music_cover/backends/svc_contentvec.py'),
          'preparation_code_sha256':sha256(ROOT/'src/music_cover/backends/svc_train.py')}
    if destination.exists():
        m=prepared(c)
        if m['identity']!=identity(spec): raise ValueError('Dataset identity changed; use a new dataset_dir')
    else:
        temporary=destination.with_name(destination.name+'.preparing')
        temporary.mkdir(parents=True,exist_ok=False)
        clips=[]
        import soundfile as sf
        import numpy as np
        for index,row in enumerate(original['clips']):
            if row['split']=='prompt': continue
            rel=f"{row['split']}/{c['voice_id']}/{index:04d}.wav"; target=temporary/rel;target.parent.mkdir(parents=True,exist_ok=True)
            audio,sr=sf.read(path(c['source_dataset'])/row['path'],dtype='float32',always_2d=True)
            if sr!=44100 or audio.shape[1]!=1 or not np.isfinite(audio).all() or np.max(np.abs(audio))>=1:
                raise ValueError('SVC requires finite unclipped 44.1 kHz mono data')
            sf.write(target,audio,sr,subtype='PCM_16')
            clips.append({**row,'source_path':row['path'],'path':rel,'sha256':sha256(target),'subtype':'PCM_16'})
        prepare_features(temporary,clips,run/'prepare_runtime')
        for split in ('train','validation','test'):
            (temporary/(split+'.txt')).write_text(''.join(str(destination/x['path'])+'\n' for x in clips if x['split']==split))
        artifacts=[{'path':str(f.relative_to(temporary)),'sha256':sha256(f)} for f in sorted(temporary.rglob('*')) if f.is_file()]
        m={**spec,'identity':identity(spec),'clips':clips,'artifacts':artifacts,'schema_version':1}
        write_json(temporary/'manifest.json',m);temporary.rename(destination)
    config=json.loads((ROOT/'third_party/so-vits-svc/configs_template/config_template.json').read_text())
    config['spk']={c['voice_id']:0};config['model']['n_speakers']=1
    config['data'].update(training_files=str(destination/'train.txt'),validation_files=str(destination/'validation.txt'))
    config['train'].update(batch_size=c['training']['batch_size'],learning_rate=c['training']['learning_rate'],all_in_mem=True)
    check_model_config(config,c['voice_id'])
    write_json(run/'model_config.json',config);write_json(out/'manifest.json',m);write_json(out/'model_config.json',config)


def train(c,run,out,args):
    from .backends.svc_train import train_model
    m=prepared(c);check_assets(training=True)
    train_model(c,run,out,m,resume=args.resume)


def validate(c,run,out,args):
    from .backends.svc_train import validate_model
    m=prepared(c);check_assets()
    validate_model(c,run,out,m,args.checkpoint,args.evaluation_split)


def export(c,run,out,args):
    import torch
    m=prepared(c);lock=check_assets()
    state=json.loads((run/'state.json').read_text())
    report_path=Path(state['stages']['02_validate']['output'])/'validation.json';report=json.loads(report_path.read_text())
    if report['dataset_identity']!=m['identity'] or report['evaluation_split']!='validation':
        raise ValueError('Export requires validation selection for this dataset')
    selected=next((r for r in report['candidates'] if r['checkpoint']==report['selected_checkpoint']),None)
    if not selected or not selected['eligible'] or sha256(selected['checkpoint'])!=selected['checkpoint_sha256']:
        raise ValueError('No unchanged technically eligible checkpoint')
    payload=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    destination=path(c.get('export_dir',f"models/voices/{c['voice_id']}/so_vits_svc/{c['version']}"))
    if destination.exists(): raise ValueError('Export version already exists; choose a new version')
    temp=destination.with_name(destination.name+'.exporting');temp.mkdir(parents=True,exist_ok=False)
    torch.save({'model':payload['model'],'iteration':payload['step'],'learning_rate':c['training']['learning_rate'],
                'optimizer':None},temp/'model.pth')
    shutil.copy2(run/'model_config.json',temp/'config.json');shutil.copy2(report_path,temp/'validation.json')
    profile={'schema_version':1,'backend':'so_vits_svc','adaptation_scope':'speaker','voice_id':c['voice_id'],
        'version':c['version'],'speaker_name':c['voice_id'],'upstream_commit':UPSTREAM_COMMIT,
        'validation_status':'candidate_requires_listening','reference_required':False,
        'selected_step':payload['step'],'dataset_identity':m['identity'],'training_run':str(run),
        'subjective_quality':'awaiting_human_review',
        'inference_assets':{a['path']:a['sha256'] for a in lock['assets'] if a.get('scope')!='training'},
        'base_assets':{a['path']:a['sha256'] for a in lock['assets'] if a.get('scope')=='training'}}
    for k,name in [('checkpoint','model.pth'),('config','config.json'),('validation','validation.json')]:
        profile[k]={'path':name,'sha256':sha256(temp/name)}
    write_json(temp/'voice.json',profile);temp.rename(destination)
    write_json(out/'export.json',{'voice_profile':str(destination/'voice.json'),**profile})


STAGES={'00_prepare':prepare,'01_train':train,'02_validate':validate,'03_export':export}

def main():
    p=argparse.ArgumentParser(description='SVC 4.1 singer adaptation in the existing base environment')
    p.add_argument('--stage',required=True,choices=STAGES);p.add_argument('--config',required=True);p.add_argument('--run',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-steps',type=int);p.add_argument('--checkpoint',action='append')
    p.add_argument('--evaluation-split',choices=['validation','test'],default='validation');a=p.parse_args();c=load_config(a.config)
    if a.max_steps is not None:
        if a.max_steps<1:p.error('--max-steps must be positive')
        c['training']['max_steps']=a.max_steps
    os.environ.update(CUDA_VISIBLE_DEVICES=str(c.get('gpu',0)),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    run=path(a.run);run.mkdir(parents=True,exist_ok=True)
    with (run/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=run/'state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'schema_version':1,'backend':'so_vits_svc','stages':{}}
        if state.get('backend')!='so_vits_svc':raise ValueError('Not an SVC training run')
        previous={'01_train':'00_prepare','02_validate':'01_train','03_export':'02_validate'}.get(a.stage)
        if previous and state['stages'].get(previous,{}).get('status')!='succeeded':raise ValueError('Run '+previous+' first')
        immutable={k:v for k,v in c.items() if k not in ('training','validation')}
        if (run/'identity.json').exists() and json.loads((run/'identity.json').read_text())!=immutable:
            raise ValueError('Run dataset/profile configuration changed; use a new run')
        write_json(run/'identity.json',immutable)
        base=run/'stages'/a.stage;base.mkdir(parents=True,exist_ok=True)
        number=max([int(x.name[8:]) for x in base.glob('attempt_*')],default=0)+1
        attempt=base/f'attempt_{number:03d}';out=attempt/'artifacts';out.mkdir(parents=True)
        write_json(attempt/'config.json',c);write_json(attempt/'command.json',{'argv':sys.argv,'cwd':str(Path.cwd())})
        state['stages'][a.stage]={'status':'running','output':str(out)};write_json(state_path,state)
        from .engine import Tee
        try:
            with (attempt/'process.log').open('a') as log, contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
                STAGES[a.stage](c,run,out,a)
            state['stages'][a.stage]['status']='succeeded'
            write_json(attempt/'outputs.json',{'status':'succeeded','artifacts':{str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file()}})
        except BaseException as error:
            state['stages'][a.stage].update(status='failed',error=str(error))
            write_json(attempt/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise
        finally:write_json(state_path,state)
        print(a.stage+': '+str(out),flush=True)

if __name__=='__main__':main()
