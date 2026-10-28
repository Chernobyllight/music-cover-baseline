"""Lightweight voice-profile resolution. Training packages are never imported here."""
import json
from pathlib import Path
import yaml
from .audio import sha256
from .common.paths import ROOT

BASE_CHECKPOINT = 'models/Seed-VC/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth'
BASE_CONFIG = 'configs/seed_svc.local.yml'


def resolve_voice(mode='zero_shot', profile=None, root=ROOT):
    root = Path(root)
    if mode not in {'zero_shot', 'finetuned'}:
        raise ValueError('voice mode must be zero_shot or finetuned')
    result = {'mode': mode, 'backend': 'seed_vc'}
    if mode == 'zero_shot':
        result.update(checkpoint=root / BASE_CHECKPOINT, config=root / BASE_CONFIG)
    else:
        if not profile:
            raise ValueError('finetuned requires --voice-profile; training is never started implicitly')
        path = (root / Path(profile).expanduser()).resolve()
        data = json.loads(path.read_text())
        if data.get('schema_version') != 1 or data.get('backend') != 'seed_vc':
            raise ValueError('Unsupported voice profile schema/backend')
        if data.get('adaptation_scope') not in (None, 'speaker'):
            raise ValueError('Only speaker-level voice profiles are supported')
        result.update(profile=path, voice_id=data['voice_id'], version=data['version'],
                      validation_status=data.get('validation_status', 'unverified'))
        for key in ('checkpoint', 'config', 'validation'):
            item = data[key]
            file = (path.parent / item['path']).resolve()
            if not file.is_file() or sha256(file) != item['sha256']:
                raise ValueError(f'Voice profile {key} missing or checksum mismatch: {file}')
            result[key] = file
    for key in ('checkpoint', 'config'):
        if not result[key].is_file():
            raise FileNotFoundError(result[key])
    config = yaml.safe_load(result['config'].read_text())
    if config['preprocess_params']['sr'] != 44100 or not config['model_params']['DiT'].get('f0_condition'):
        raise ValueError('This singing pipeline requires a 44.1 kHz F0-conditioned Seed-VC model')
    return result


def voice_record(resolved):
    result = {k: v for k, v in resolved.items() if not isinstance(v, Path)}
    for key in ('checkpoint', 'config', 'profile', 'validation'):
        if key in resolved:
            result[key] = {'path': str(resolved[key]), 'sha256': sha256(resolved[key])}
    return result
