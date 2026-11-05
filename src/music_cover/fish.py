"""Fish Diffusion contracts. No neural-network imports in the pipeline controller."""
import json
import math
import subprocess
from pathlib import Path
from .audio import sha256
from .common.paths import ROOT
from .common.voice_profiles import verified_files

UPSTREAM_COMMIT = '8e8f8cd849717f2db561e6c194df1a320c910e62'
DEFAULT_SETTINGS = {'sampler_interval': 10, 'noise_predictor': 'plms',
                    'chunk_seconds': 8.0, 'overlap_seconds': 0.2, 'pad_seconds': 0.5}
MODEL_SPEC = {'architecture': 'diff_svc_v2', 'sample_rate': 44100, 'hop_length': 512,
              'mel_channels': 128, 'content_dim': 256, 'content_layer': 9,
              'content_projection': True, 'f0_predictor': 'parselmouth',
              'f0_min': 40.0, 'f0_max': 2000.0, 'keep_zeros': False,
              'timesteps': 1000, 'speaker_count': 1, 'pitch_encoding': 'log',
              'vocoder_log': 'base10'}


def validate_settings(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_SETTINGS):
        raise ValueError('Unknown Fish Diffusion inference settings')
    p = {**DEFAULT_SETTINGS, **value}
    if type(p['sampler_interval']) is not int or not 1 <= p['sampler_interval'] <= 100:
        raise ValueError('sampler_interval must be an integer in 1..100')
    if p['noise_predictor'] not in {'plms', 'unipc'}:
        raise ValueError('Fish noise_predictor must be plms or unipc')
    for key in ('chunk_seconds', 'overlap_seconds', 'pad_seconds'):
        if type(p[key]) not in (int, float) or not math.isfinite(p[key]):
            raise ValueError('Invalid Fish '+key)
    if not 2 <= p['chunk_seconds'] <= 30:
        raise ValueError('chunk_seconds must be 2..30')
    if not 0 < p['overlap_seconds'] < p['chunk_seconds']/2:
        raise ValueError('Invalid Fish overlap_seconds')
    if not .1 <= p['pad_seconds'] <= 2:
        raise ValueError('pad_seconds must be .1..2')
    return p


def check_model_config(config):
    if config.get('model') != MODEL_SPEC:
        raise ValueError('Fish model/feature/vocoder architecture differs from the supported pretrained bundle')
    if not isinstance(config.get('speaker_name'), str) or not config['speaker_name']:
        raise ValueError('Missing Fish speaker_name')


def check_assets(root=ROOT, training=False):
    root = Path(root)
    lock = json.loads((root/'configs/fish_diffusion_assets.lock.json').read_text())
    if lock['upstream_commit'] != UPSTREAM_COMMIT:
        raise ValueError('Fish upstream lock differs')
    upstream = root/'third_party/fish-diffusion'
    actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise ValueError('Fish checkout revision differs from lock')
    patches = json.loads((root/'patches/fish_diffusion/files.json').read_text())
    dirty = subprocess.check_output(['git', '-C', str(upstream), 'diff', '--name-only', 'HEAD'], text=True).splitlines()
    if set(dirty) != set(patches):
        raise ValueError('Fish checkout must have exactly the documented base compatibility patch')
    for name, item in patches.items():
        if sha256(upstream/name) != item['patched_sha256']:
            raise ValueError('Fish compatibility source changed: '+name)
    for item in lock['assets']:
        if item.get('scope') == 'training' and not training:
            continue
        f = root/item['path']
        if not f.is_file() or f.stat().st_size != item['bytes'] or sha256(f) != item['sha256']:
            raise ValueError('Fish asset missing/corrupt: '+str(f))
    return lock


def resolve_fish_voice(mode, profile, root=ROOT):
    root = Path(root)
    if mode != 'finetuned' or not profile:
        raise ValueError('fish_diffusion requires an exported finetuned voice profile')
    p = (root/Path(profile).expanduser()).resolve()
    data = json.loads(p.read_text())
    if (data.get('schema_version') != 1 or data.get('backend') != 'fish_diffusion'
            or data.get('adaptation_scope') != 'speaker' or data.get('upstream_commit') != UPSTREAM_COMMIT):
        raise ValueError('Expected a compatible speaker-level Fish Diffusion profile')
    lock = json.loads((root/'configs/fish_diffusion_assets.lock.json').read_text())
    assets = {a['path']: a['sha256'] for a in lock['assets'] if a.get('scope') != 'training'}
    if data.get('inference_assets') != assets:
        raise ValueError('Fish profile content encoder/vocoder assets differ')
    result = {'backend': 'fish_diffusion', 'mode': mode, 'profile': p,
              'voice_id': data['voice_id'], 'version': data['version'],
              'speaker_name': data['speaker_name'], 'reference_required': False,
              'validation_status': data.get('validation_status', 'unverified'),
              'reference_independence': data.get('reference_independence', 'not_inference_condition; independence_requires_dataset_check'),
              'upstream_commit': UPSTREAM_COMMIT}
    result.update(verified_files(p, data, ('checkpoint', 'config', 'validation'), label='Fish profile'))
    check_model_config(json.loads(result['config'].read_text()))
    return result


def checked_state(model, saved, replace_speaker=False):
    """All acoustic weights must match. Only speaker reinitialization is allowed."""
    current = model.state_dict()
    allowed = {'speaker_encoder.embedding.weight'} if replace_speaker else set()
    missing, extra = set(current)-set(saved), set(saved)-set(current)
    mismatch = [k for k in set(current)&set(saved) if k not in allowed and current[k].shape != saved[k].shape]
    if missing-allowed or extra or mismatch:
        raise ValueError(f'Fish state mismatch: missing={sorted(missing-allowed)[:8]}, extra={sorted(extra)[:8]}, shapes={mismatch[:8]}')
    values = {k: (current[k] if k in allowed else saved[k]) for k in current}
    model.load_state_dict(values, strict=True)
    return {'loaded_tensors': len(current)-len(allowed), 'initialized_tensors': sorted(allowed),
            'coverage': 'all non-speaker acoustic tensors matched exactly'}
