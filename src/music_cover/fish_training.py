"""Independent Fish prepare/features/train/validate/export with portable stage state."""
import argparse
import contextlib
import fcntl
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
import yaml
from .audio import sha256
from .common.paths import ROOT
from .common.io import write_json, identity
from .common.voice_paths import portable_path, training_output
from .fish import MODEL_SPEC, UPSTREAM_COMMIT, check_assets
from .fish_config import validate_config, checked_dataset


def path(value): return (ROOT/Path(value).expanduser()).resolve()


def checked_manifest(c):
    m = json.loads((path(c['dataset_dir'])/'manifest.json').read_text())
    if m['source_manifest_sha256'] != c['source_manifest_sha256'] or m['voice_id'] != c['voice_id']:
        raise ValueError('Prepared Fish dataset identity differs')
    source = path(c.get('source_dataset') or c['recordings_dir'])/'manifest.json'
    if sha256(source) != c['source_manifest_sha256']: raise ValueError('Source manifest changed')
    for row in m['clips']:
        f = path(c['dataset_dir'])/row['path']
        if not f.resolve().is_relative_to(path(c['dataset_dir'])) or sha256(f) != row['sha256']:
            raise ValueError('Prepared clip changed: '+str(f))
    return m


def prepare(c,run,out,args):
    import numpy as np
    import soundfile as sf
    destination = path(c['dataset_dir'])
    source = path(c.get('source_dataset') or c['recordings_dir'])
    if sha256(source/'manifest.json') != c['source_manifest_sha256']:
        raise ValueError('Source manifest changed since configuration generation')
    spec = {'voice_id':c['voice_id'],'source_manifest_sha256':c['source_manifest_sha256'],
            'segment_seconds':c['segment_seconds'],'format':'44100-mono-FLOAT',
            'preparation_code_sha256':sha256(Path(__file__))}
    if destination.exists():
        m = checked_manifest(c)
        if m['identity'] != identity(spec): raise ValueError('Prepared dataset parameters changed; use a new dataset-dir')
    else:
        temp = destination.with_name(destination.name+'.preparing')
        temp.mkdir(parents=True,exist_ok=False)
        clips = []
        if 'source_dataset' in c:
            original = checked_dataset(source,c['voice_id'])
            sources = [{**r,'file':source/r['path']} for r in original['clips'] if r['split'] != 'prompt']
        else:
            sources = [{**r,'file':path(r['path']),'group':r['id']} for r in c['sources']]
        for row in sources:
            if sha256(row['file']) != row['sha256']: raise ValueError('Source audio changed')
            audio,sr = sf.read(row['file'],dtype='float32',always_2d=True)
            audio = audio.mean(axis=1)
            if sr != 44100 or not np.isfinite(audio).all(): raise ValueError('Expected finite 44.1 kHz source audio')
            width = round(c['segment_seconds']*sr)
            for first in range(0,len(audio),width):
                y = audio[first:first+width]
                if len(y)<sr: continue
                peak = float(np.max(np.abs(y)))
                gain = min(1.,.99/max(peak,1e-12))
                rel = f"{row['split']}/{len(clips):05d}.wav"
                target=temp/rel;target.parent.mkdir(parents=True,exist_ok=True)
                sf.write(target,y*gain,sr,subtype='FLOAT')
                clips.append({'path':rel,'split':row['split'],'group':row['group'],
                    'sha256':sha256(target),'source_sha256':row['sha256'],
                    'source_path':portable_path(row['file']),'offset_seconds':first/sr,
                    'duration_s':len(y)/sr,'peak_gain':gain})
        if {r['split'] for r in clips} != {'train','validation','test'}:
            raise ValueError('Empty dataset split after cutting')
        m = {**spec,'schema_version':1,'identity':identity(spec),'clips':clips}
        write_json(temp/'manifest.json',m);temp.rename(destination)
    write_json(out/'manifest.json',m)
    write_json(out/'dataset.json',{'path':portable_path(destination),'identity':m['identity']})


def features(c,run,out,args):
    from .backends.fish_train import extract_features
    m = checked_manifest(c)
    lock = check_assets(training=True)
    extract_features(c,m,lock,out)


def feature_manifest(c,run):
    m = checked_manifest(c)
    report = json.loads((training_output(run,'01_features')/'features.json').read_text())
    if report['dataset_identity'] != m['identity'] or report['model'] != c['model'] or report['augmentation'] != c['augmentation']:
        raise ValueError('Fish features no longer match dataset/configuration')
    lock = check_assets(training=True)
    if report['assets'] != {a['path']:a['sha256'] for a in lock['assets'] if a['scope']!='training'}:
        raise ValueError('Fish feature assets changed')
    for row in report['features']:
        f = path(c['dataset_dir'])/row['path']
        if not f.resolve().is_relative_to(path(c['dataset_dir'])) or sha256(f)!=row['sha256']:
            raise ValueError('Feature file changed: '+str(f))
    return report


def train(c,run,out,args):
    from .backends.fish_train import train_model
    train_model(c,run,out,feature_manifest(c,run),resume=args.resume)


def validate(c,run,out,args):
    from .backends.fish_train import validate_model
    report = feature_manifest(c,run)
    train_report=json.loads((training_output(run,'02_train')/'training.json').read_text())
    checkpoint = path(args.checkpoint) if args.checkpoint else run/train_report['checkpoint']
    if not args.checkpoint and sha256(checkpoint)!=train_report['checkpoint_sha256']:
        raise ValueError('Final checkpoint changed')
    validate_model(c,run,out,report,checkpoint,args.evaluation_split)


def export(c,run,out,args):
    from .backends.fish_diffusion import runtime
    torch=runtime()
    features=feature_manifest(c,run);lock=check_assets()
    report_file=training_output(run,'03_validate')/'validation.json'
    report=json.loads(report_file.read_text())
    if report['features_identity']!=features['identity'] or report['evaluation_split']!='validation':
        raise ValueError('Export requires validation for this feature dataset')
    checkpoint=path(report['checkpoint'])
    if sha256(checkpoint)!=report['checkpoint_sha256']: raise ValueError('Validated checkpoint changed')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['features_identity']!=features['identity']: raise ValueError('Checkpoint dataset identity differs')
    destination=path(c['export_dir'])
    if destination.exists(): raise ValueError('Export already exists; use a new version')
    temp=destination.with_name(destination.name+'.exporting');temp.mkdir(parents=True,exist_ok=False)
    torch.save({'model':payload['model'],'step':payload['step'],'features_identity':payload['features_identity']},temp/'model.ckpt')
    write_json(temp/'config.json',{'model':MODEL_SPEC,'speaker_name':c['voice_id']})
    shutil.copy2(report_file,temp/'validation.json')
    profile={'schema_version':1,'backend':'fish_diffusion','adaptation_scope':'speaker',
        'voice_id':c['voice_id'],'speaker_name':c['voice_id'],'version':c['version'],
        'upstream_commit':UPSTREAM_COMMIT,'selected_step':payload['step'],
        'training_run':portable_path(run),'features_identity':features['identity'],
        'reference_required':False,'reference_independence':'not_inference_condition; independence_requires_dataset_check',
        'validation_status':'candidate_requires_listening' if report['eligible'] else 'quality_review_required',
        'assessment_status':'passed' if report['eligible'] else 'review_required',
        'subjective_quality':'awaiting_human_review','recommended_settings':c['inference'],
        'inference_assets':{a['path']:a['sha256'] for a in lock['assets'] if a['scope']!='training'}}
    for key,name in [('checkpoint','model.ckpt'),('config','config.json'),('validation','validation.json')]:
        profile[key]={'path':name,'sha256':sha256(temp/name)}
    write_json(temp/'voice.json',profile);temp.rename(destination)
    write_json(out/'export.json',{'voice_profile':portable_path(destination/'voice.json'),**profile})


STAGES={'00_prepare':prepare,'01_features':features,'02_train':train,'03_validate':validate,'04_export':export}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=STAGES);p.add_argument('--config',required=True);p.add_argument('--run',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--checkpoint')
    p.add_argument('--max-steps',type=int);p.add_argument('--evaluation-split',choices=['validation','test'],default='validation')
    a=p.parse_args(argv);c=yaml.safe_load(path(a.config).read_text())
    if a.max_steps is not None:c['training']['max_steps']=a.max_steps
    validate_config(c)
    if Path(sys.prefix).resolve()!=Path('/opt/conda').resolve(): raise RuntimeError('Fish training requires base /opt/conda/bin/python')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(c['gpu']),WANDB_MODE='disabled',HF_HUB_OFFLINE='1')
    run=path(a.run);run.mkdir(parents=True,exist_ok=True)
    with (run/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=run/'state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {'schema_version':1,'backend':'fish_diffusion','stages':{}}
        if state.get('backend')!='fish_diffusion':raise ValueError('Not a Fish training run')
        order=list(STAGES);index=order.index(a.stage)
        if index and state['stages'].get(order[index-1],{}).get('status')!='succeeded':
            raise ValueError('Run '+order[index-1]+' first')
        immutable={k:v for k,v in c.items() if k not in ('training','validation','inference')}
        if (run/'identity.json').exists() and json.loads((run/'identity.json').read_text())!=immutable:
            raise ValueError('Run data/model identity changed; use a new run')
        write_json(run/'identity.json',immutable)
        base=run/'stages'/a.stage;base.mkdir(parents=True,exist_ok=True)
        number=max([int(x.name[8:]) for x in base.glob('attempt_*')],default=0)+1
        attempt=base/f'attempt_{number:03d}';out=attempt/'artifacts';out.mkdir(parents=True)
        write_json(attempt/'config.json',c)
        write_json(attempt/'command.json',{'argv':sys.argv,'cwd':str(Path.cwd()),'python':sys.executable,'prefix':sys.prefix})
        for later in order[index+1:]:
            if later in state['stages']:state['stages'][later]['status']='invalidated'
        state['stages'][a.stage]={'status':'running','output':str(out.relative_to(run))};write_json(state_path,state)
        from .engine import Tee
        try:
            with (attempt/'process.log').open('a') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
                STAGES[a.stage](c,run,out,a)
            state['stages'][a.stage]['status']='succeeded'
            write_json(attempt/'outputs.json',{'status':'succeeded','artifacts':{str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file()}})
        except BaseException as error:
            state['stages'][a.stage].update(status='failed',error=str(error))
            write_json(attempt/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise
        finally:write_json(state_path,state)
    print(a.stage+': '+str(out),flush=True)


if __name__=='__main__':main()
