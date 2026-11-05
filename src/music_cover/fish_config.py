"""Explicit, fresh Fish Diffusion training and inference configuration generation."""
import argparse
import json
import math
import sys
from collections import Counter
import soundfile as sf
import yaml
from .audio import sha256
from .common.paths import ROOT, read_run_state
from .common.voice_paths import portable_path
from .common.voice_config import (project_path, check_id, check_splits, checked_recordings,
    fresh_targets, stage_artifacts, validation_windows, write_new_config)
from .fish import DEFAULT_SETTINGS, MODEL_SPEC, validate_settings, resolve_fish_voice


def checked_dataset(folder, voice_id):
    folder = folder.resolve()
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest.get('voice_id') != voice_id or manifest.get('smoke_test'):
        raise ValueError('Require a matching train/validation/test singer dataset')
    groups, hashes, paths = {}, {}, set()
    for row in manifest['clips']:
        p = (folder/row['path']).resolve()
        split = row['split']
        if split not in {'train','validation','test','prompt'}:
            raise ValueError('Unknown dataset split')
        if not p.is_relative_to(folder) or p in paths or sha256(p) != row['sha256']:
            raise ValueError('Dataset path/hash changed or repeated')
        paths.add(p)
        if split == 'prompt': continue
        for table, key in ((groups,row['group']), (hashes,row['sha256'])):
            if key in table and table[key] != split: raise ValueError('Dataset split leakage')
            table[key] = split
        info = sf.info(str(p))
        if info.duration < 1 or info.samplerate != 44100 or info.channels != 1:
            raise ValueError('Require clean mono 44.1 kHz clips of at least one second')
    if set(groups.values()) != {'train','validation','test'}:
        raise ValueError('Need nonempty train/validation/test recording groups')
    return manifest


def validate_config(c):
    for key in ('voice_id','version'): check_id(c[key],key)
    if c.get('backend') != 'fish_diffusion': raise ValueError('Expected a Fish training config')
    if c.get('model') != MODEL_SPEC: raise ValueError('Unsupported Fish model/feature specification')
    if type(c['gpu']) is not int or c['gpu'] < 0: raise ValueError('gpu must be nonnegative')
    t = c['training']
    for key in ('max_steps','save_every','batch_size','crop_frames','warmup_steps','decay_steps'):
        if type(t[key]) is not int or t[key] < 1: raise ValueError('Invalid '+key)
    if t['crop_frames'] < 64: raise ValueError('crop_frames must be >= 64')
    if type(t['seed']) is not int or not 0 <= t['seed'] < 2**32: raise ValueError('Invalid seed')
    for key in ('learning_rate','start_learning_rate','final_learning_rate'):
        if not math.isfinite(t[key]) or not 0 < t[key] <= .001: raise ValueError('Invalid '+key)
    if t['decay_steps'] <= t['warmup_steps']: raise ValueError('decay_steps must exceed warmup_steps')
    if t['precision'] not in {'float32','bfloat16'}: raise ValueError('Invalid precision')
    if not 1 <= c['segment_seconds'] <= 10: raise ValueError('segment_seconds must be 1..10')
    if type(c['augmentation']['pitch_shift_probability']) not in (int,float) or not 0 <= c['augmentation']['pitch_shift_probability'] <= 3:
        raise ValueError('Invalid pitch shift probability')
    validate_settings(c['inference'])
    return c


def training_advisories(training):
    """Explain shortened schedules without blocking explicit smoke tests."""
    if training['max_steps'] <= training['warmup_steps']:
        return ['Training ends during/at warmup; this is a smoke-test schedule, not evidence of voice quality.']
    if training['max_steps'] < training['decay_steps']:
        return ['Training ends before the configured decay endpoint; compare held-out audio before choosing this checkpoint.']
    return []


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='kind', required=True)
    t = sub.add_parser('training')
    for name in ('voice-id','version','source-run'): t.add_argument('--'+name,required=True)
    source = t.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-dataset'); source.add_argument('--recordings-dir')
    t.add_argument('--validation-ids', nargs='+', default=[]); t.add_argument('--test-ids', nargs='+', default=[])
    t.add_argument('--validation-windows', nargs='+', required=True)
    t.add_argument('--dataset-dir'); t.add_argument('--export-dir')
    for name,default in [('max-steps',5000),('save-every',1000),('batch-size',4),('crop-frames',384),
                         ('warmup-steps',1000),('decay-steps',5000),('gpu',0)]:
        t.add_argument('--'+name,type=int,default=default)
    for name,default in [('learning-rate',4e-4),('start-learning-rate',1e-5),('final-learning-rate',2e-5),
                         ('segment-seconds',8),('pitch-shift-probability',1.5)]:
        t.add_argument('--'+name,type=float,default=default)
    t.add_argument('--precision',choices=['float32','bfloat16'],default='bfloat16')
    i = sub.add_parser('inference'); i.add_argument('--voice-profile',required=True)
    for name,default in DEFAULT_SETTINGS.items():
        i.add_argument('--'+name.replace('_','-'),type=type(default))
    for item in (t,i):
        item.add_argument('--seed',type=int,default=42);item.add_argument('--output');item.add_argument('--plan',action='store_true')
    return p


def training_config(a):
    for name in ('voice_id','version'): check_id(getattr(a,name),name)
    path = lambda x: project_path(x,ROOT)
    portable = lambda x: portable_path(x,ROOT)
    song = path(a.source_run)
    state = read_run_state(song,root=ROOT)
    ref = stage_artifacts(state,'00_prepare',['reference'])['reference']
    stems = stage_artifacts(state,'04_vocal_separation',['vocals','instrumental'])
    durations = [sf.info(str(f)).duration for f in stems.values()]
    if abs(durations[0]-durations[1]) > .25: raise ValueError('Stage 04 stem lengths differ')
    windows = validation_windows(a.validation_windows,min(durations))
    name = f'{a.voice_id}_{a.version}'
    output = path(a.output or f'configs/voice_training/fish_diffusion/{name}.yaml')
    dataset = path(a.dataset_dir or f'data/voice_datasets/{a.voice_id}/fish_diffusion/{a.version}')
    export = path(a.export_dir or f'models/voices/{a.voice_id}/fish_diffusion/{a.version}')
    source = path(a.source_dataset or a.recordings_dir)
    fresh_targets({'output':output,'dataset-dir':dataset,'export-dir':export},[source,song])
    c = {'backend':'fish_diffusion','voice_id':a.voice_id,'version':a.version,'gpu':a.gpu,
         'dataset_dir':portable(dataset),'export_dir':portable(export),'reference':portable(ref),
         'model':dict(MODEL_SPEC),'segment_seconds':a.segment_seconds,
         'training':{k:getattr(a,k) for k in ('max_steps','save_every','batch_size','crop_frames','warmup_steps',
                      'decay_steps','learning_rate','start_learning_rate','final_learning_rate','seed','precision')},
         'augmentation':{'pitch_shift_probability':a.pitch_shift_probability,'key_shifts':[-5.0,5.0]},
         'inference':dict(DEFAULT_SETTINGS),
         'validation':{'source_run':portable(song),'windows':windows,'seed':a.seed,
             'source_files':{k:{'path':portable(v),'sha256':sha256(v)} for k,v in {'reference':ref,**stems}.items()}}}
    if a.source_dataset:
        m = checked_dataset(source,a.voice_id)
        c['source_dataset'] = portable(source)
        counts = dict(Counter(x['split'] for x in m['clips']))
    else:
        check_splits(a.validation_ids,a.test_ids)
        recordings,excluded,_ = checked_recordings(a,ROOT)
        c.update(recordings_dir=portable(source),sources=recordings,excluded_recordings=excluded)
        counts = dict(Counter(x['split'] for x in recordings))
    c['source_manifest_sha256'] = sha256(source/'manifest.json')
    return validate_config(c), output, {'split_counts':counts,'training_advisories':training_advisories(c['training'])}


def inference_config(a):
    if not 0 <= a.seed < 2**32: raise ValueError('Invalid seed')
    profile = project_path(a.voice_profile,ROOT)
    voice = resolve_fish_voice('finetuned',profile)
    data = json.loads(profile.read_text())
    settings = validate_settings({**data.get('recommended_settings',{}),
        **{k:getattr(a,k) for k in DEFAULT_SETTINGS if getattr(a,k) is not None}})
    output = project_path(a.output or f"configs/voice_inference/fish_diffusion/{voice['voice_id']}_{voice['version']}.yaml",ROOT)
    fresh_targets({'output':output},[profile.parent])
    c = {'modules':{'05_voice_conversion':{'backend':'fish_diffusion','mode':'finetuned',
         'voice_profile':portable_path(profile,ROOT),'candidate_count':1,'candidate_seeds':[a.seed],
         'fish':settings,'svc':{},'rvc':{}}}}
    return c,output,{'profile_sha256':sha256(profile)}


def main(argv=None):
    p = parser(); a = p.parse_args(argv)
    try:
        c,output,summary = training_config(a) if a.kind == 'training' else inference_config(a)
        print(json.dumps({'python':sys.executable,'output':str(output),**summary},indent=2))
        print(yaml.safe_dump(c,sort_keys=False,allow_unicode=True))
        if a.plan: print('PLAN: no files created')
        else: write_new_config(output,c); print('Created '+str(output))
    except (ValueError,KeyError,TypeError,OSError) as error: p.error(str(error))


if __name__ == '__main__': main()
