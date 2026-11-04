"""Independent, resumable RVC prepare/features/train/index/validate/export stages."""
import argparse
import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
import yaml
from .common.paths import ROOT, read_run_state
from .common.resources import python_for
from .common.io import write_json,identity
from .common.voice_paths import portable_path
from .common.rvc_paths import portable_config
from .audio import sha256,decode
from .rvc import check_assets,check_config,UPSTREAM_COMMIT


def path(value):return (ROOT/Path(value).expanduser()).resolve()


def portable(value):return portable_path(path(value),ROOT)


def portable_job(job):
    result=dict(job)
    for key in ('source','converted','output','checkpoint'):
        if key in result:result[key]=portable(result[key])
    if 'references' in result:result['references']=[portable(p) for p in result['references']]
    return result


def source_manifest(c):
    folder=path(c['source_dataset']);m=json.loads((folder/'manifest.json').read_text());groups={};hashes={}
    if m['voice_id']!=c['voice_id']:raise ValueError('RVC source voice differs')
    for row in m['clips']:
        f=(folder/row['path']).resolve()
        if not f.is_relative_to(folder) or not f.is_file() or sha256(f)!=row['sha256']:raise ValueError('RVC source audio changed')
        for table,key in [(groups,row['group']),(hashes,row['sha256'])]:
            if key in table and table[key]!=row['split']:raise ValueError('RVC train/evaluation leakage')
            table[key]=row['split']
    return m


def prepared(c):
    folder=path(c['dataset_dir']);m=json.loads((folder/'manifest.json').read_text())
    if m['source_manifest_sha256']!=sha256(path(c['source_dataset'])/'manifest.json'):raise ValueError('RVC source manifest changed')
    for row in m['clips']:
        if sha256(folder/row['path'])!=row['sha256']:raise ValueError('Prepared RVC audio changed')
    return m


def prepare(c,run,out,args):
    import librosa
    import soundfile as sf
    import numpy as np
    source=source_manifest(c);folder=path(c['dataset_dir'])
    spec={'source_manifest_sha256':sha256(path(c['source_dataset'])/'manifest.json'),'voice_id':c['voice_id'],
          'sample_rate':48000,'format':'FLOAT','policy':'original group splits; prompt excluded; no augmentation',
          'upstream_commit':UPSTREAM_COMMIT,'assets':json.loads((ROOT/'configs/rvc_assets.lock.json').read_text())}
    if folder.exists():
        m=prepared(c)
        if m['identity']!=identity(spec):raise ValueError('RVC dataset identity differs')
    else:
        temp=folder.with_name(folder.name+'.partial');temp.mkdir(parents=True,exist_ok=False);rows=[]
        for i,row in enumerate(source['clips']):
            if row['split']=='prompt':continue
            src=path(c['source_dataset'])/row['path'];audio,sr=sf.read(src,dtype='float32',always_2d=True);audio=audio.mean(1)
            audio=librosa.resample(audio,orig_sr=sr,target_sr=48000)
            if not np.isfinite(audio).all():raise ValueError('Invalid training audio')
            rel=f"{row['split']}/{i:04d}.wav";target=temp/rel;target.parent.mkdir(exist_ok=True)
            sf.write(target,audio,48000,subtype='FLOAT')
            rows.append({**row,'source_path':portable(src),'source_sha256':row['sha256'],'path':rel,'sha256':sha256(target),'sample_rate':48000,'frames':len(audio),'duration_s':len(audio)/48000,'subtype':'FLOAT'})
        m={**spec,'identity':identity(spec),'source_path_base':'project','clips':rows};write_json(temp/'manifest.json',m);temp.rename(folder)
    config=json.loads((ROOT/'third_party/rvc/configs/v2/48k.json').read_text())
    config.update(version='v2',f0=1,f0_policy='unvoiced_zero',speaker_id=0)
    config['model']['spk_embed_dim']=1;config['train']['seed']=c['training']['seed']
    check_config(config);write_json(run/'model_config.json',config)
    write_json(out/'dataset.json',{'path_base':'project','dataset_dir':portable(folder),'identity':m['identity'],'counts':{s:sum(r['split']==s for r in m['clips']) for s in ('train','validation','test')}})


def features(c,run,out,args):
    from .backends.rvc_train import features as extract
    write_json(out/'features.json',extract(path(c['dataset_dir']),prepared(c)))


def train(c,run,out,args):
    from .backends.rvc_train import train as fit
    fit(c,run,out,prepared(c),args.resume)


def build_index(c,run,out,args):
    import faiss
    import numpy as np
    folder=path(c['dataset_dir']);m=prepared(c);feature_manifest=json.loads((folder/'features.json').read_text())
    expected={x['path']:x['sha256'] for x in feature_manifest['artifacts']};arrays=[];sources=[]
    for row in m['clips']:
        if row['split']!='train':continue
        f=folder/'features'/f"{Path(row['path']).stem}.content.npy"
        if sha256(f)!=expected[str(f.relative_to(folder))]:raise ValueError('RVC indexed feature changed')
        a=np.load(f,allow_pickle=False)
        if a.ndim!=2 or a.shape[1]!=768 or not np.isfinite(a).all():raise ValueError('Invalid RVC index features')
        arrays.append(a);sources.append({'path':portable(f),'sha256':sha256(f),'group':row['group'],'split':'train','vectors':len(a)})
    vectors=np.ascontiguousarray(np.concatenate(arrays),dtype='float32');faiss.omp_set_num_threads(4)
    index=faiss.IndexFlatL2(768);index.add(vectors)
    index_path=run/'added.index';faiss.write_index(index,str(index_path))
    report={'path_base':'project','dataset_identity':m['identity'],'split':'train','dimension':768,'vectors':len(vectors),'type':'IndexFlatL2','k':8,
            'sources':sources,'index_sha256':sha256(index_path),'features_manifest_sha256':sha256(folder/'features.json')}
    write_json(run/'index_manifest.json',report);write_json(out/'index.json',report)


def validation_checkpoint(c,run,requested=None):
    return path(requested) if requested else run/'checkpoints'/f"model_{c['training']['max_steps']:08d}.pth"


def validate(c,run,out,args):
    from .backends.rvc import runtime,Converter
    import torch
    m=prepared(c);refs=[str(path(c['dataset_dir'])/r['path']) for r in m['clips'] if r['split']=='validation']
    candidates=[validation_checkpoint(c,run,args.checkpoint)]
    rates=[c.get('inference',{}).get('index_rate',.5)]
    config=json.loads((run/'model_config.json').read_text());jobs=[]
    for ci,case in enumerate(c['validation']['cases']):
        state=read_run_state(path(case['source_run']),root=ROOT);item=state['stages']['04_vocal_separation']['outputs']['vocals']
        if sha256(item['path'])!=item['sha256']:raise ValueError('RVC validation source changed')
        source=out/f'source_{ci:03d}.wav';decode(item['path'],source,rate=44100,channels=1,start=case['start'],duration=case['duration'])
    with runtime():
        for checkpoint in candidates:
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if payload['dataset_identity']!=m['identity']:raise ValueError('RVC checkpoint dataset differs')
            evaluated_step=payload.get('step')
            del payload
            converter=Converter(checkpoint,config,run/'added.index')
            for rate in rates:
                for ci,case in enumerate(c['validation']['cases']):
                    folder=out/checkpoint.stem/f'index_{rate:g}'/f'case_{ci:03d}';folder.mkdir(parents=True)
                    settings={**c.get('inference',{}),'index_rate':rate};source=out/f'source_{ci:03d}.wav'
                    result=converter.infer(source,folder/'converted.wav',settings,seed=42);write_json(folder/'conversion.json',result)
                    jobs.append({'source':str(source),'converted':str(folder/'converted.wav'),'references':refs,'output':str(folder/'metrics'),
                        'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'case':ci,'index_rate':rate})
            del converter;torch.cuda.empty_cache()
    write_json(out/'metrics_request.json',{'path_base':'project','jobs':[portable_job(job) for job in jobs]})
    write_json(out/'runtime_metrics_request.json',{'jobs':jobs})
    command=[str(python_for(ROOT, 'voice_metrics')),'-m','music_cover.backends.cli','svc_metrics','--request',str(out/'runtime_metrics_request.json'),'--output',str(out)]
    write_json(out/'metrics_command.json',{'argv':command,'python':command[0],'cwd':str(ROOT),'controller_python':sys.executable})
    with (out/'metrics.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
    rows=[portable_job(row) for row in json.loads((out/'metrics.json').read_text())['cases']];summaries=[]
    write_json(out/'metrics.json',{'path_base':'project','cases':rows})
    for checkpoint in candidates:
        for rate in rates:
            cases=[r for r in rows if path(r['checkpoint'])==checkpoint and r['index_rate']==rate]
            summaries.append({'checkpoint':portable(checkpoint),'checkpoint_sha256':sha256(checkpoint),'index_rate':rate,'cases':cases,
                'eligible':len(cases)==len(c['validation']['cases']) and all(r['eligible'] for r in cases),
                'mean_ecapa':sum(r['metrics']['speaker_similarity'] for r in cases)/len(cases)})
    selected=summaries[0]
    report={'path_base':'project','dataset_identity':m['identity'],'evaluation_split':'validation','references':[portable(p) for p in refs],'candidates':summaries,'selected':selected,
            'selection_rule':'one requested checkpoint, defaulting to configured final training step; fixed inference preset; validation without checkpoint ranking',
            'recommended_checkpoint':selected['checkpoint'] if selected['eligible'] else None,
            'assessment_status':'passed' if selected['eligible'] else 'review_required', 'blocks_export':False,
            'evaluated_step':evaluated_step,
            'configured_final_step':c['training']['max_steps'],
            'subjective_quality':'awaiting_human_review'}
    write_json(out/'validation.json',report);write_json(run/'validation.json',report)


def export(c,run,out,args):
    m=prepared(c);report=json.loads((run/'validation.json').read_text());selected=report['selected']
    if report['evaluation_split']!='validation' or report['dataset_identity']!=m['identity'] or not selected:
        raise ValueError('RVC export requires a validation-selected model for this dataset')
    checkpoint=path(selected['checkpoint'])
    if sha256(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('Selected RVC model changed')
    if selected['index_rate']!=c.get('inference',{}).get('index_rate',.5):raise ValueError('RVC validation inference preset differs')
    destination=path(c['export_dir']);temp=destination.with_name(destination.name+'.partial')
    if destination.exists():raise ValueError('RVC export version already exists')
    temp.mkdir(parents=True,exist_ok=False)
    files={'checkpoint':(checkpoint,'model.pth'),'config':(run/'model_config.json','config.json'),
           'index':(run/'added.index','added.index'),'index_manifest':(run/'index_manifest.json','index_manifest.json'),
           'validation':(run/'validation.json','validation.json')}
    lock=json.loads((ROOT/'configs/rvc_assets.lock.json').read_text())
    profile={'schema_version':1,'backend':'rvc','adaptation_scope':'speaker','voice_id':c['voice_id'],'version':c['version'],
        'speaker_id':0,'reference_required':False,'upstream_commit':UPSTREAM_COMMIT,'dataset_identity':m['identity'],'training_run':portable(run),'path_base':'project',
        'validation_status':'objective_validation_passed_requires_listening' if selected['eligible'] else 'quality_review_required',
        'assessment_status':'passed' if selected['eligible'] else 'review_required', 'subjective_quality':'awaiting_human_review',
        'recommended_settings':{**c.get('inference',{}),'index_rate':selected['index_rate']},
        'inference_assets':{a['path']:a['sha256'] for a in lock['assets'] if a['scope']=='inference'}}
    for k,(src,name) in files.items():shutil.copy2(src,temp/name);profile[k]={'path':name,'sha256':sha256(temp/name)}
    write_json(temp/'voice.json',profile);temp.rename(destination);write_json(out/'export.json',{'path_base':'project','voice_profile':portable(destination/'voice.json')})


STAGES={'00_prepare':prepare,'01_features':features,'02_train':train,'03_index':build_index,'04_validate':validate,'05_export':export}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--config',required=True);p.add_argument('--run',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-steps',type=int);p.add_argument('--checkpoint',help='Optional inference checkpoint to validate; defaults to configured final step');a=p.parse_args()
    c=portable_config(yaml.safe_load(path(a.config).read_text()),ROOT)
    if a.max_steps is not None:c['training']['max_steps']=a.max_steps
    if c['training']['max_steps']<1:raise ValueError('max_steps must be positive')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(c.get('gpu',0)),RVC_CUDA_GRAPH='0')
    check_assets(training=True);run=path(a.run);run.mkdir(parents=True,exist_ok=True)
    with (run/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state_path=run/'state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {'schema_version':1,'backend':'rvc','stages':{}}
        if state['backend']!='rvc':raise ValueError('Not an RVC training run')
        dependencies={'01_features':['00_prepare'],'02_train':['01_features'],'03_index':['01_features'],'04_validate':['02_train','03_index'],'05_export':['04_validate']}
        if any(state['stages'].get(s,{}).get('status')!='succeeded' for s in dependencies.get(a.stage,[])):raise ValueError('RVC prerequisite stage incomplete')
        immutable={k:v for k,v in c.items() if k not in ('training','validation','inference')}
        if (run/'identity.json').exists() and json.loads((run/'identity.json').read_text())!=immutable:raise ValueError('RVC run identity changed')
        write_json(run/'identity.json',immutable);base=run/'stages'/a.stage;base.mkdir(parents=True,exist_ok=True)
        number=max([int(x.name[8:]) for x in base.glob('attempt_*')],default=0)+1;attempt=base/f'attempt_{number:03d}';out=attempt/'artifacts';out.mkdir(parents=True)
        write_json(attempt/'config.json',c);write_json(attempt/'command.json',{'argv':sys.argv,'cwd':str(Path.cwd()),'python':sys.executable,'prefix':sys.prefix})
        state['stages'][a.stage]={'status':'running','output':out.relative_to(run).as_posix()};write_json(state_path,state)
        from .engine import Tee
        try:
            with (attempt/'process.log').open('a') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):STAGES[a.stage](c,run,out,a)
            state['stages'][a.stage]['status']='succeeded';write_json(attempt/'outputs.json',{'status':'succeeded','files':{str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file()}})
        except BaseException as error:
            state['stages'][a.stage].update(status='failed',error=str(error));write_json(attempt/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise
        finally:write_json(state_path,state)
        print(a.stage+': '+str(out),flush=True)


if __name__=='__main__':main()
