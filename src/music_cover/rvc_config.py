"""Generate fresh RVC training or inference configuration; never train or infer."""
import argparse
from collections import Counter
import json
import math
import sys
import soundfile as sf
import yaml
from .audio import sha256
from .common.paths import ROOT, read_run_state
from .common.voice_paths import portable_path
from .common.voice_config import (project_path, check_id, fresh_targets, read_json,
                                 stage_artifacts, validation_windows, write_new_config)
from .rvc import DEFAULT_SETTINGS, validate_settings, resolve_rvc_voice
from .voice_candidates import candidate_seeds


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='kind', required=True)
    train = sub.add_parser('training')
    for name in ('voice-id', 'version', 'source-dataset', 'source-run'):
        train.add_argument('--' + name, required=True)
    train.add_argument('--validation-windows', nargs='+', required=True, metavar='START:DURATION')
    train.add_argument('--dataset-dir'); train.add_argument('--export-dir')
    for name, value in [('max-steps', 4000), ('save-every', 1000), ('batch-size', 6),
                        ('input-crop-frames', 400)]:
        train.add_argument('--' + name, type=int, default=value)
    train.add_argument('--learning-rate', type=float, default=0.0001)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--gpu', type=int, default=0)
    infer = sub.add_parser('inference')
    infer.add_argument('--voice-profile', required=True)
    infer.add_argument('--candidate-count', type=int, default=1)
    infer.add_argument('--seed', type=int, default=42)
    infer.add_argument('--candidate-seeds', type=int, nargs='+')
    for item in (train, infer):
        item.add_argument('--output')
        item.add_argument('--plan', action='store_true')
        for name in DEFAULT_SETTINGS:
            item.add_argument('--' + name.replace('_', '-'),
                              type=str if name == 'f0_predictor' else float)
    return p


def settings(a, defaults=None):
    overrides = {k: getattr(a, k) for k in DEFAULT_SETTINGS if getattr(a, k) is not None}
    return validate_settings({**(defaults or {}), **overrides})


def checked_dataset(folder, voice_id):
    m = read_json(folder/'manifest.json')
    if m.get('voice_id') != voice_id or m.get('smoke_test', False):
        raise ValueError('Expected a formal dataset for the requested voice-id')
    if not isinstance(m.get('clips'), list) or not m['clips']:
        raise ValueError('Dataset needs a nonempty clips list')
    groups, hashes, paths = {}, {}, set()
    for row in m['clips']:
        split, group = row['split'], row['group']
        if split not in {'train', 'validation', 'test', 'prompt'} or not isinstance(group, str) or not group:
            raise ValueError('Invalid dataset split/group')
        f = (folder/row['path']).resolve()
        if not f.is_relative_to(folder) or f in paths or not f.is_file() or sha256(f) != row['sha256']:
            raise ValueError('Dataset audio is missing, duplicated or changed')
        paths.add(f)
        for table, key in [(groups, group), (hashes, row['sha256'])]:
            if key in table and table[key] != split:
                raise ValueError('Dataset train/validation/test leakage')
            table[key] = split
        info = sf.info(str(f))
        if info.duration < 1 or info.channels != 1:
            raise ValueError('Dataset clips must be mono and at least one second')
    counts = Counter(r['split'] for r in m['clips'])
    if not {'train', 'validation', 'test'} <= set(counts):
        raise ValueError('Dataset needs train, validation and test clips')
    return m, dict(counts)


def training_config(a):
    check_id(a.voice_id, 'voice-id'); check_id(a.version, 'version')
    if a.gpu < 0 or not 0 <= a.seed < 2**32:
        raise ValueError('Invalid GPU or seed')
    for name in ('max_steps', 'save_every', 'batch_size', 'input_crop_frames'):
        if getattr(a, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if not math.isfinite(a.learning_rate) or not 0 < a.learning_rate <= 0.001:
        raise ValueError('learning-rate must be in (0, 0.001]')
    path = lambda value: project_path(value, ROOT)
    portable = lambda value: portable_path(value, ROOT)
    source = path(a.source_dataset); song = path(a.source_run)
    output = path(a.output or f'configs/voice_training/rvc/{a.voice_id}_{a.version}.yaml')
    dataset = path(a.dataset_dir or f'data/voice_datasets/{a.voice_id}/rvc/{a.version}')
    export = path(a.export_dir or f'models/voices/{a.voice_id}/rvc/{a.version}')
    fresh_targets({'output': output, 'dataset-dir': dataset, 'export-dir': export}, [source, song])
    manifest, counts = checked_dataset(source, a.voice_id)
    stems = stage_artifacts(read_run_state(song, root=ROOT), '04_vocal_separation', ['vocals', 'instrumental'])
    lengths = [sf.info(str(f)).duration for f in stems.values()]
    if abs(lengths[0]-lengths[1]) > 0.25:
        raise ValueError('Stage 04 vocals/instrumental durations differ')
    windows = validation_windows(a.validation_windows, min(lengths))
    c = {'voice_id': a.voice_id, 'version': a.version, 'gpu': a.gpu,
         'source_dataset': portable(source), 'dataset_dir': portable(dataset), 'export_dir': portable(export),
         'training': {k: getattr(a, k) for k in ('max_steps', 'save_every', 'batch_size', 'learning_rate', 'seed', 'input_crop_frames')},
         'inference': settings(a),
         'validation': {'cases': [{'source_run': portable(song), **w} for w in windows]},
         'configuration_generation': {'tool': 'music_cover.rvc_config',
             'source_manifest_sha256': sha256(source/'manifest.json'),
             'checked_stage_artifacts': {k: {'path': portable(f), 'sha256': sha256(f)} for k, f in stems.items()}}}
    return c, output, {'clips': counts, 'dataset_dir': str(dataset), 'export_dir': str(export),
                       'training': c['training'], 'inference': c['inference']}


def inference_config(a):
    path = lambda value: project_path(value, ROOT)
    profile = path(a.voice_profile)
    voice = resolve_rvc_voice('finetuned', profile, ROOT)
    check_id(voice['voice_id'], 'profile voice_id'); check_id(voice['version'], 'profile version')
    data = read_json(profile)
    output = path(a.output or f"configs/voice_inference/rvc/{voice['voice_id']}_{voice['version']}.yaml")
    fresh_targets({'output': output}, [profile.parent])
    if not 0 <= a.seed < 2**32:
        raise ValueError('seed must be in [0, 2**32)')
    seeds = candidate_seeds(a.seed, a.candidate_count, a.candidate_seeds)
    preset = settings(a, data.get('recommended_settings', {}))
    c = {'modules': {'05_voice_conversion': {
        'backend': 'rvc', 'mode': 'finetuned', 'voice_profile': portable_path(profile, ROOT),
        'candidate_count': a.candidate_count, 'candidate_seeds': seeds, 'svc': {}, 'rvc': preset}}}
    return c, output, {'voice_profile': str(profile), 'profile_sha256': sha256(profile),
                       'settings_source': 'RVC defaults < exported recommended_settings < explicit CLI options',
                       'candidate_seeds': seeds, 'inference': preset}


def main(argv=None):
    p = parser(); a = p.parse_args(argv)
    try:
        config, output, summary = (training_config(a) if a.kind == 'training' else inference_config(a))
        print(json.dumps({'python': sys.executable, 'output': str(output), **summary}, ensure_ascii=False, indent=2))
        print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        if a.plan:
            print('PLAN: checks passed; no files created.')
        else:
            write_new_config(output, config)
            print(f'Created RVC {a.kind} config: {output}')
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
