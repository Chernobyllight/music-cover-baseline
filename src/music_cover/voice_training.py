"""Explicit prepare/train/validate/export workflow; never called by conversion."""
from __future__ import annotations
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
from .audio import decode, inspect_audio, sha256
from .common.io import write_json, identity
from .common.paths import ROOT
from .voice import BASE_CHECKPOINT, BASE_CONFIG


def path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def load_config(filename):
    c = yaml.safe_load(path(filename).read_text())
    if not isinstance(c, dict): raise ValueError('Training config must be a mapping')
    for key in ('voice_id', 'version', 'dataset_dir', 'sources', 'reference', 'training', 'validation'):
        if key not in c: raise ValueError(f'Missing training config key: {key}')
    if not c['sources']: raise ValueError('Provide target-singer recordings in sources before preparing')
    if not isinstance(c.get('smoke_test', False), bool): raise ValueError('smoke_test must be boolean')
    if not 1 <= float(c.get('segment_seconds', 8)) <= 30: raise ValueError('segment_seconds must be 1..30')
    t = c['training']
    for key in ('max_steps', 'save_every', 'batch_size'):
        if not isinstance(t[key], int) or isinstance(t[key], bool) or t[key] <= 0: raise ValueError(f'{key} must be positive')
    t['learning_rate'] = float(t.get('learning_rate', 1e-5))
    if not 0 < t['learning_rate'] <= 1e-3: raise ValueError('Invalid learning_rate')
    return c


def dataset_manifest(c):
    m = json.loads((path(c['dataset_dir']) / 'manifest.json').read_text())
    if bool(m['smoke_test']) != bool(c.get('smoke_test')) or m['voice_id'] != c['voice_id']:
        raise ValueError('Dataset voice/smoke status differs from config; prepare a new version')
    if m['reference_sha256'] != sha256(path(c['reference'])):
        raise ValueError('Reference changed after dataset preparation')
    for row in m['clips']:
        p = path(c['dataset_dir']) / row['path']
        if not p.is_file() or sha256(p) != row['sha256']: raise ValueError(f'Dataset clip changed: {p}')
    return m


def prepare(c, run, out, args):
    import numpy as np
    import soundfile as sf
    destination = path(c['dataset_dir'])
    records, hashes, groups = [], {}, {}
    for item in c['sources']:
        split, group, source = item['split'], item['group'], path(item['path'])
        if split not in {'train', 'validation', 'test', 'prompt'}: raise ValueError('Invalid dataset split')
        digest = sha256(source)
        if groups.get(group, split) != split or hashes.get(digest, split) != split:
            raise ValueError('Recording/group leakage between dataset splits')
        groups[group], hashes[digest] = split, split
        records.append({**item, 'path': str(source), 'sha256': digest})
    if 'train' not in groups.values(): raise ValueError('Training recordings are required')
    if not c.get('smoke_test', False):
        if not {'validation', 'test'} <= set(groups.values()):
            raise ValueError('Formal training needs separate validation/test recording groups; smoke_test is explicit')
        if sha256(path(c['reference'])) in {x['sha256'] for x in records if x['split'] != 'prompt'}:
            raise ValueError('Inference prompt must be held out from train/validation/test recordings')
    spec = {'recordings': records, 'segment_seconds': c.get('segment_seconds', 8), 'voice_id': c['voice_id'],
            'reference_sha256': sha256(path(c['reference'])), 'smoke_test': bool(c.get('smoke_test', False))}
    key = identity(spec)
    if destination.exists():
        old = dataset_manifest(c)
        if old['identity'] != key: raise ValueError('Dataset version already exists with different inputs; use a new dataset_dir')
        write_json(out / 'manifest.json', old)
        return
    temporary = destination.with_name(destination.name + '.preparing')
    if temporary.exists(): raise ValueError(f'Incomplete preparation exists; inspect before retrying: {temporary}')
    temporary.mkdir(parents=True)
    clips, rejected = [], []
    for index, row in enumerate(records):
        decoded = temporary / f'decoded_{index}.wav'
        decode(row['path'], decoded, rate=44100, channels=1)
        audio, sr = sf.read(decoded, dtype='float32')
        width = round(float(c.get('segment_seconds', 8)) * sr)
        for number, first in enumerate(range(0, len(audio), width)):
            block = audio[first:first + width]
            if len(block) < sr or np.sqrt(np.mean(block ** 2)) < 1e-4:
                rejected.append({'recording': index, 'start_s': first / sr, 'reason': 'short_or_silent'})
                continue
            rel = f"{row['split']}/{index:04d}_{number:04d}.wav"
            target = temporary / rel
            target.parent.mkdir(exist_ok=True)
            sf.write(target, block, sr, subtype='FLOAT')
            info = inspect_audio(target)
            clips.append({'path': rel, 'split': row['split'], 'group': row['group'], 'recording': index,
                          'start_s': first / sr, 'clipped_fraction': float(np.mean(np.abs(block) >= .999)), **info})
        decoded.unlink()
    required = {'train'} if c.get('smoke_test') else {'train', 'validation', 'test'}
    if not required <= {x['split'] for x in clips}: raise ValueError('At least one required split has no usable clips')
    manifest = {**spec, 'identity': key, 'schema_version': 1, 'clips': clips, 'rejected': rejected,
                'assessment': 'integration_only' if c.get('smoke_test') else 'held_out_recordings',
                'quality_note': 'Duration/finite/RMS/clipping checks; clean solo singing still needs listening review.'}
    write_json(temporary / 'manifest.json', manifest)
    temporary.rename(destination)
    write_json(out / 'manifest.json', manifest)


def train(c, run, out, args):
    from .backends.voice_train import train_voice
    dataset_manifest(c)
    train_voice(c, run, out, resume=args.resume)


def validate(c, run, out, args):
    from .backends.voice_validate import validate_voice
    dataset_manifest(c)
    validate_voice(c, run, out, args.checkpoint, args.evaluation_split)


def export(c, run, out, args):
    m = dataset_manifest(c)
    state = json.loads((run / 'state.json').read_text())
    validation_path = Path(state['stages']['02_validate']['output']) / 'validation.json'
    report = json.loads(validation_path.read_text())
    if report['dataset_identity'] != m['identity']: raise ValueError('Validation dataset differs')
    checkpoint = path(args.checkpoint[0]) if args.checkpoint else Path(report['selected_checkpoint'])
    row = next((x for x in report['candidates'] if Path(x['checkpoint']) == checkpoint), None)
    if row is None or row['checkpoint_sha256'] != sha256(checkpoint): raise ValueError('Selected checkpoint was not validated or has changed')
    if not row['eligible']: raise ValueError('Checkpoint failed technical validation; export refused')
    version = path(c.get('export_dir', f"models/voices/{c['voice_id']}/seed_vc/{c['version']}"))
    if version.exists(): raise ValueError('Exported voice versions are immutable; choose a new version/export_dir')
    temporary = version.with_name(version.name + '.exporting')
    temporary.mkdir(parents=True, exist_ok=False)
    import torch
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    torch.save({'net': payload['net']}, temporary / 'model.pth')
    shutil.copy2(run / 'model_config.yml', temporary / 'config.yml')
    shutil.copy2(validation_path, temporary / 'validation.json')
    profile = {'schema_version': 1, 'backend': 'seed_vc', 'voice_id': c['voice_id'], 'version': c['version'],
               'validation_status': 'integration_only' if c.get('smoke_test') else 'candidate_requires_listening',
               'subjective_quality': 'awaiting_human_review', 'reference_required': True,
               'dataset_identity': m['identity'], 'training_run': str(run),
               'validation': {'path': 'validation.json', 'sha256': sha256(temporary / 'validation.json')},
               'base_checkpoint_sha256': sha256(path(c.get('base_checkpoint', BASE_CHECKPOINT))),
               'checkpoint': {'path': 'model.pth', 'sha256': sha256(temporary / 'model.pth')},
               'config': {'path': 'config.yml', 'sha256': sha256(temporary / 'config.yml')}}
    write_json(temporary / 'voice.json', profile)
    temporary.rename(version)
    write_json(out / 'export.json', {'voice_profile': str(version / 'voice.json'), **profile})


STAGES = {'00_prepare': prepare, '01_train': train, '02_validate': validate, '03_export': export}


def main():
    parser = argparse.ArgumentParser(description='Seed-VC target-voice adaptation; independent of song inference')
    parser.add_argument('--stage', required=True, choices=STAGES)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--resume', action='store_true', help='Restore optimizer, scheduler, step and RNG state')
    parser.add_argument('--max-steps', type=int, help='Total step budget, including resumed steps')
    parser.add_argument('--checkpoint', action='append', help='Candidate checkpoint; repeat for comparison')
    parser.add_argument('--evaluation-split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    c = load_config(args.config)
    if args.max_steps is not None:
        if args.max_steps < 1: parser.error('--max-steps must be positive')
        c['training']['max_steps'] = args.max_steps
    os.environ['CUDA_VISIBLE_DEVICES'] = str(c.get('gpu', 0))
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    run = path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = run / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'schema_version': 1, 'stages': {}}
        previous = {'01_train': '00_prepare', '02_validate': '01_train', '03_export': '02_validate'}.get(args.stage)
        if previous and state['stages'].get(previous, {}).get('status') != 'succeeded':
            # Interrupted training may be resumed from its durable checkpoint.
            raise ValueError(f'Run {previous} successfully first')
        base = run / 'stages' / args.stage
        base.mkdir(parents=True, exist_ok=True)
        numbers = [int(x.name[8:]) for x in base.glob('attempt_*') if x.name[8:].isdigit()]
        attempt = base / f'attempt_{max(numbers, default=0) + 1:03d}'
        out = attempt / 'artifacts'
        out.mkdir(parents=True)
        write_json(attempt / 'config.json', c)
        write_json(attempt / 'command.json', {'argv': sys.argv, 'cwd': str(Path.cwd())})
        state['stages'][args.stage] = {'status': 'running', 'output': str(out)}
        write_json(state_path, state)
        from .engine import Tee
        try:
            with (attempt / 'process.log').open('a') as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                STAGES[args.stage](c, run, out, args)
            artifacts = {str(p.relative_to(out)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                         for p in out.rglob('*') if p.is_file()}
            write_json(attempt / 'outputs.json', {'status': 'succeeded', 'artifacts': artifacts})
            state['stages'][args.stage].update(status='succeeded')
        except BaseException as error:
            state['stages'][args.stage].update(status='failed', error=str(error))
            write_json(attempt / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
            raise
        finally:
            write_json(state_path, state)
        print(f'{args.stage}: {out}', flush=True)


if __name__ == '__main__': main()
