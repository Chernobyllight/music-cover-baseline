"""RVC contracts, with no training or GPU imports in the controller."""
import json
import math
import subprocess
from pathlib import Path
from .audio import sha256
from .common.paths import ROOT
from .common.voice_profiles import verified_files

UPSTREAM_COMMIT = '81eed5e8f68b6bed1789f682fe78cdd324495afc'
DEFAULT_SETTINGS = {'f0_predictor': 'rmvpe', 'index_rate': .5, 'protect': .33,
                    'chunk_seconds': 8., 'overlap_seconds': .2, 'pad_seconds': .5}


def validate_settings(value):
    if not isinstance(value, dict) or set(value)-set(DEFAULT_SETTINGS):
        raise ValueError('Unknown RVC settings; expected an rvc mapping')
    p = {**DEFAULT_SETTINGS, **value}
    if p['f0_predictor'] != 'rmvpe': raise ValueError('RVC requires RMVPE')
    for k in set(p)-{'f0_predictor'}:
        if isinstance(p[k], bool) or not isinstance(p[k], (int, float)) or not math.isfinite(p[k]):
            raise ValueError('Invalid RVC '+k)
    if not 0 <= p['index_rate'] <= 1: raise ValueError('index_rate must be 0..1')
    if not 0 <= p['protect'] <= .5: raise ValueError('protect must be 0..0.5')
    if not 2 <= p['chunk_seconds'] <= 30: raise ValueError('chunk_seconds must be 2..30')
    if not 0 < p['overlap_seconds'] < p['chunk_seconds']/2: raise ValueError('Invalid RVC overlap')
    if not .1 <= p['pad_seconds'] <= 2: raise ValueError('Invalid RVC padding')
    return p


def check_config(c):
    if c.get('version') != 'v2' or c.get('f0') != 1 or c.get('f0_policy') != 'unvoiced_zero':
        raise ValueError('RVC requires v2 / F0 / unvoiced_zero policy')
    if c['data']['sampling_rate'] != 48000 or c['data']['hop_length'] != 480:
        raise ValueError('RVC requires matched 48 kHz / hop 480 architecture')
    if c['model']['spk_embed_dim'] != 1 or c.get('speaker_id') != 0:
        raise ValueError('RVC currently requires one speaker, ID 0')


def check_assets(root=ROOT, training=False):
    root = Path(root); lock = json.loads((root/'configs/rvc_assets.lock.json').read_text())
    if lock['upstream_commit'] != UPSTREAM_COMMIT: raise ValueError('RVC lock revision differs')
    upstream = root/'third_party/rvc'
    actual = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if actual != UPSTREAM_COMMIT: raise ValueError('RVC checkout revision differs')
    if subprocess.check_output(['git','-C',str(upstream),'diff','HEAD','--'],text=True):
        raise ValueError('RVC upstream tracked files changed')
    for a in lock['assets']:
        if a['scope'] == 'training' and not training: continue
        f = root/a['path']
        if not f.is_file() or f.stat().st_size != a['bytes'] or sha256(f) != a['sha256']:
            raise ValueError('RVC asset missing/corrupt: '+str(f))
    return lock


def resolve_rvc_voice(mode, profile, root=ROOT):
    root = Path(root)
    if mode != 'finetuned': raise ValueError('RVC requires finetuned mode')
    if not profile: raise ValueError('RVC requires an exported voice_profile')
    p = (root/Path(profile).expanduser()).resolve(); data = json.loads(p.read_text())
    if data.get('schema_version') != 1 or data.get('backend') != 'rvc' or data.get('adaptation_scope') != 'speaker':
        raise ValueError('Expected a speaker-level RVC profile')
    if data.get('upstream_commit') != UPSTREAM_COMMIT: raise ValueError('RVC profile revision differs')
    lock = json.loads((root/'configs/rvc_assets.lock.json').read_text())
    expected = {a['path']:a['sha256'] for a in lock['assets'] if a['scope']=='inference'}
    if data.get('inference_assets') != expected: raise ValueError('RVC encoder/F0 assets differ')
    result = {'backend':'rvc','mode':mode,'profile':p,'voice_id':data['voice_id'],'version':data['version'],
              'speaker_id':0,'reference_required':False,'validation_status':data.get('validation_status','unverified')}
    result.update(verified_files(p, data, ('checkpoint', 'config', 'index', 'index_manifest', 'validation'),
                                 label='RVC profile', contained=True))
    check_config(json.loads(result['config'].read_text()))
    manifest=json.loads(result['index_manifest'].read_text())
    if manifest.get('dataset_identity')!=data.get('dataset_identity') or manifest.get('split')!='train' or manifest.get('dimension')!=768:
        raise ValueError('RVC index dataset/split/dimension mismatch')
    if manifest.get('index_sha256')!=data['index']['sha256']:
        raise ValueError('RVC index identity mismatch')
    return result
