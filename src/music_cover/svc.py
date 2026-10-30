"""Lightweight SVC contracts; importing this module does not import torch/fairseq."""
import json
import math
from pathlib import Path
from .audio import sha256
from .common.paths import ROOT
from .common.voice_profiles import verified_files

UPSTREAM_COMMIT = '730930d337d171479eadf305f96cbed4bb393e77'
DEFAULT_SETTINGS = {'f0_predictor': 'rmvpe', 'noise_scale': 0.4, 'chunk_seconds': 20.0,
                    'overlap_seconds': 0.5, 'pad_seconds': 0.5}


def validate_settings(value):
    if not isinstance(value, dict): raise ValueError('svc must be a mapping')
    if set(value) - set(DEFAULT_SETTINGS): raise ValueError('Unknown SVC settings')
    p = {**DEFAULT_SETTINGS, **value}
    if p['f0_predictor'] != 'rmvpe': raise ValueError('Only RMVPE is validated for this SVC integration')
    for k in ('noise_scale', 'chunk_seconds', 'overlap_seconds', 'pad_seconds'):
        if isinstance(p[k], bool) or not isinstance(p[k], (int, float)) or not math.isfinite(p[k]):
            raise ValueError(f'Invalid SVC {k}')
    if not 0 <= p['noise_scale'] <= 1: raise ValueError('noise_scale must be 0..1')
    if not 2 <= p['chunk_seconds'] <= 60: raise ValueError('chunk_seconds must be 2..60')
    if not 0 < p['overlap_seconds'] < p['chunk_seconds'] / 2: raise ValueError('Invalid overlap_seconds')
    if not .1 <= p['pad_seconds'] <= 2: raise ValueError('pad_seconds must be .1..2')
    return p


def check_model_config(c, speaker=None):
    if c['data']['sampling_rate'] != 44100 or c['data']['hop_length'] != 512:
        raise ValueError('SVC requires the validated 44.1 kHz / hop 512 architecture')
    m = c['model']
    dimensions = {'vec768l12': 768, 'vec256l9': 256}
    dimension = dimensions.get(m['speech_encoder'])
    if dimension is None or m['ssl_dim'] != dimension or m['gin_channels'] != dimension:
        raise ValueError('SVC requires a matched vec768l12/768 or vec256l9/256 architecture')
    if m.get('vol_embedding') or m.get('use_depthwise_conv') or m.get('use_transformer_flow') or m.get('flow_share_parameter'):
        raise ValueError('This integration supports the standard non-volume SVC architecture')
    if m.get('vocoder_name') != 'nsf-hifigan': raise ValueError('Unsupported SVC vocoder')
    if speaker is not None and (speaker not in c['spk'] or not 0 <= c['spk'][speaker] < m['n_speakers']):
        raise ValueError('Speaker missing from SVC config')


def check_assets(root=ROOT, training=False):
    root = Path(root)
    lock = json.loads((root/'configs/so_vits_svc_assets.lock.json').read_text())
    if lock['upstream_commit'] != UPSTREAM_COMMIT: raise ValueError('Unexpected SVC revision')
    import subprocess
    actual = subprocess.check_output(['git', '-C', str(root/'third_party/so-vits-svc'), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != UPSTREAM_COMMIT: raise ValueError('SVC checkout revision differs from lock')
    dirty = subprocess.check_output(['git', '-C', str(root/'third_party/so-vits-svc'), 'diff', 'HEAD', '--'], text=True)
    if dirty: raise ValueError('SVC upstream checkout has tracked changes')
    for a in lock['assets']:
        if a.get('scope') == 'training' and not training: continue
        f = root/a['path']
        if not f.is_file() or f.stat().st_size != a['bytes'] or sha256(f) != a['sha256']:
            raise ValueError(f'SVC asset missing/corrupt: {f}')
    return lock


def resolve_svc_voice(mode, profile, root=ROOT):
    root = Path(root)
    if mode != 'finetuned': raise ValueError('so_vits_svc does not support zero_shot')
    if not profile: raise ValueError('so_vits_svc requires an exported --voice-profile')
    p = (root/Path(profile).expanduser()).resolve()
    data = json.loads(p.read_text())
    if data.get('schema_version') != 1 or data.get('backend') != 'so_vits_svc' or data.get('adaptation_scope') != 'speaker':
        raise ValueError('Expected a speaker-level so_vits_svc profile')
    if data.get('upstream_commit') != UPSTREAM_COMMIT: raise ValueError('SVC profile revision mismatch')
    lock = json.loads((root/'configs/so_vits_svc_assets.lock.json').read_text())
    assets = {a['path']: a['sha256'] for a in lock['assets'] if a.get('scope') != 'training'}
    if data.get('inference_assets') != assets: raise ValueError('SVC profile encoder/F0 assets differ')
    result = {'backend': 'so_vits_svc', 'mode': mode, 'profile': p, 'voice_id': data['voice_id'],
              'version': data['version'], 'speaker_name': data['speaker_name'],
              'validation_status': data.get('validation_status', 'unverified'),
              'reference_required': False, 'upstream_commit': UPSTREAM_COMMIT}
    result.update(verified_files(p, data, ('checkpoint', 'config', 'validation'), label='SVC profile'))
    check_model_config(json.loads(result['config'].read_text()), result['speaker_name'])
    return result


def checked_state(model, saved, *, replace_speaker=False):
    """Fail on ALL unexpected/missing/mismatched weights, except explicit speaker initialization."""
    current = model.state_dict()
    allowed = {'emb_g.weight'} if replace_speaker else set()
    if (set(current) - set(saved)) - allowed or set(saved) - set(current):
        raise ValueError(f'SVC checkpoint keys differ: missing={sorted(set(current)-set(saved))[:8]}, extra={sorted(set(saved)-set(current))[:8]}')
    bad = [k for k in current if not (replace_speaker and k == 'emb_g.weight') and tuple(current[k].shape) != tuple(saved[k].shape)]
    if bad: raise ValueError('SVC checkpoint shape mismatch: '+str(bad[:8]))
    result = dict(saved)
    if replace_speaker: result['emb_g.weight'] = current['emb_g.weight']
    model.load_state_dict(result, strict=True)
    return {'loaded_tensors': len(result) - int(replace_speaker), 'initialized_tensors': ['emb_g.weight'] if replace_speaker else [],
            'coverage': 'all non-speaker parameters matched exactly'}
