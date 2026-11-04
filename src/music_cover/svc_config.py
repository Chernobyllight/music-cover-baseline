"""Create fresh SVC training/inference YAML from checked inputs; never train."""
import argparse
from collections import Counter
import json
import math
import re
import sys
import soundfile as sf
import yaml
from .audio import sha256
from .common.paths import ROOT, read_run_state
from .common.voice_paths import portable_path
from .common.voice_config import (project_path, fresh_targets, read_json,
                                 stage_artifacts, validation_windows, write_new_config)
from .svc import DEFAULT_SETTINGS, validate_settings, resolve_svc_voice


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='kind', required=True)
    train = sub.add_parser('training')
    for name in ('voice-id', 'version', 'source-dataset', 'source-run'):
        train.add_argument('--' + name, required=True)
    train.add_argument('--validation-windows', nargs='+', required=True, metavar='START:DURATION')
    train.add_argument('--dataset-dir')
    train.add_argument('--export-dir')
    for name, value in [('max-steps', 4000), ('save-every', 1000), ('batch-size', 6)]:
        train.add_argument('--' + name, type=int, default=value)
    train.add_argument('--learning-rate', type=float, default=0.0001)
    train.add_argument('--gpu', type=int, default=0)
    train.add_argument('--peak-policy', choices=['error', 'attenuate_if_needed'], default='attenuate_if_needed',
                       help='Attenuate float peaks >= 1 only when later preparing SVC PCM16 copies')
    train.add_argument('--peak-limit', type=float, default=0.99)
    infer = sub.add_parser('inference')
    infer.add_argument('--voice-profile', required=True)
    for name in DEFAULT_SETTINGS:
        infer.add_argument('--' + name.replace('_', '-'),
                           type=str if name == 'f0_predictor' else float)
    for command in (train, infer):
        command.add_argument('--seed', type=int, default=42)
        command.add_argument('--output')
        command.add_argument('--plan', action='store_true')
    return p


def check_id(value, name):
    # Match the existing SVC training loader, including its version restrictions.
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'{name} must contain only letters, digits, underscores or hyphens')


def check_seed(seed):
    if not 0 <= seed < 2**32:
        raise ValueError('seed must be in [0, 2**32)')


def checked_dataset(folder, voice_id, reference, audio_policy):
    from .svc_training import source_manifest, prepare_audio
    # Absolute inputs here let the established consumer enforce its own contract;
    # only project-relative paths are published in the generated YAML.
    manifest = source_manifest({'voice_id': voice_id, 'source_dataset': str(folder),
                                'reference': str(reference)})
    paths = set()
    attenuation = []
    for row in manifest['clips']:
        file = (folder / row['path']).resolve()
        if row['split'] not in {'train', 'validation', 'test', 'prompt'} or file in paths:
            raise ValueError('Invalid split or repeated dataset file')
        paths.add(file)
        if row['split'] == 'prompt':
            continue
        audio, rate = sf.read(file, dtype='float32', always_2d=True)
        if len(audio) < rate:
            raise ValueError(f'SVC clips must be at least one second: {file}')
        try:
            _, levels = prepare_audio(audio, rate, audio_policy)
        except ValueError as error:
            raise ValueError(f'{file}: {error}') from error
        if levels['gain'] < 1:
            attenuation.append({'path': row['path'], **levels})
    return manifest, dict(Counter(row['split'] for row in manifest['clips'])), attenuation


def training_config(a):
    check_id(a.voice_id, 'voice-id')
    check_id(a.version, 'version')
    check_seed(a.seed)
    if a.gpu < 0:
        raise ValueError('gpu must be nonnegative')
    if any(getattr(a, k) < 1 for k in ('max_steps', 'save_every', 'batch_size')):
        raise ValueError('Training steps, save interval and batch size must be positive')
    if not math.isfinite(a.learning_rate) or not 0 < a.learning_rate <= 0.001:
        raise ValueError('learning-rate must be in (0, 0.001]')
    path = lambda value: project_path(value, ROOT)
    portable = lambda value: portable_path(value, ROOT)
    source, song = path(a.source_dataset), path(a.source_run)
    output = path(a.output or f'configs/voice_training/so_vits_svc/{a.voice_id}_{a.version}.yaml')
    dataset = path(a.dataset_dir or f'data/voice_datasets/{a.voice_id}/so_vits_svc/{a.version}')
    export = path(a.export_dir or f'models/voices/{a.voice_id}/so_vits_svc/{a.version}')
    fresh_targets({'output': output, 'dataset-dir': dataset, 'export-dir': export}, [source, song])
    state = read_run_state(song, root=ROOT)
    reference = stage_artifacts(state, '00_prepare', ['reference'])['reference']
    stems = stage_artifacts(state, '04_vocal_separation', ['vocals', 'instrumental'])
    lengths = [sf.info(str(file)).duration for file in stems.values()]
    if abs(lengths[0] - lengths[1]) > 0.25:
        raise ValueError('Stage 04 vocals/instrumental durations differ')
    windows = validation_windows(a.validation_windows, min(lengths))
    from .svc_training import audio_settings
    audio_policy = audio_settings({'audio': {'peak_policy': a.peak_policy, 'peak_limit': a.peak_limit}})
    _, counts, attenuation = checked_dataset(source, a.voice_id, reference, audio_policy)
    config = {
        'voice_id': a.voice_id, 'version': a.version, 'gpu': a.gpu,
        'source_dataset': portable(source), 'dataset_dir': portable(dataset),
        'reference': portable(reference), 'export_dir': portable(export),
        'audio': audio_policy,
        'training': {k: getattr(a, k) for k in ('max_steps', 'save_every', 'batch_size', 'learning_rate', 'seed')},
        'validation': {'source_run': portable(song), 'windows': windows},
        'configuration_generation': {
            'tool': 'music_cover.svc_config', 'source_manifest_sha256': sha256(source / 'manifest.json'),
            'checked_stage_artifacts': {
                key: {'path': portable(file), 'sha256': sha256(file)}
                for key, file in {'reference': reference, **stems}.items()}}}
    return config, output, {'clips': counts, 'dataset_dir': str(dataset), 'export_dir': str(export),
                            'audio_preparation': audio_policy, 'clips_requiring_attenuation': attenuation,
                            'validation_inference_defaults': dict(DEFAULT_SETTINGS)}


def inference_config(a):
    check_seed(a.seed)
    path = lambda value: project_path(value, ROOT)
    profile = path(a.voice_profile)
    voice = resolve_svc_voice('finetuned', profile, ROOT)
    check_id(voice['voice_id'], 'profile voice_id')
    check_id(voice['version'], 'profile version')
    output = path(a.output or f"configs/voice_inference/so_vits_svc/{voice['voice_id']}_{voice['version']}.yaml")
    fresh_targets({'output': output}, [profile.parent])
    data = read_json(profile)
    defaults = validate_settings(data.get('recommended_settings', {}))
    preset = validate_settings({**defaults, **{k: getattr(a, k) for k in DEFAULT_SETTINGS if getattr(a, k) is not None}})
    config = {'modules': {'05_voice_conversion': {
        'backend': 'so_vits_svc', 'mode': 'finetuned',
        'voice_profile': portable_path(profile, ROOT), 'candidate_count': 1,
        'candidate_seeds': [a.seed], 'svc': preset, 'rvc': {}}}}
    return config, output, {'voice_profile': str(profile), 'profile_sha256': sha256(profile),
                            'candidate_seeds': [a.seed], 'inference': preset,
                            'settings_source': 'SVC defaults < exported recommended_settings < explicit CLI options'}


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    try:
        config, output, summary = (training_config(a) if a.kind == 'training' else inference_config(a))
        print(json.dumps({'python': sys.executable, 'output': str(output), **summary}, ensure_ascii=False, indent=2))
        print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        if a.plan:
            print('PLAN: checks passed; no files created.')
        else:
            write_new_config(output, config)
            print(f'Created SVC {a.kind} config: {output}')
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
