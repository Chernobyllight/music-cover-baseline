"""Backend-scoped model/runtime identities and preflight requirements."""
import json
import shutil
import subprocess
from pathlib import Path
from ..audio import sha256

ENVIRONMENTS = {'fish_diffusion': '/opt/conda', 'rvc': '/opt/conda', 'so_vits_svc': '/opt/conda', 'sheetsage': 'sheetsage2', 'yue': '/opt/conda', 'separator': '/opt/conda',
                'seed_vc': '/opt/conda', 'rmvpe': '/opt/conda', 'review_stems': 'separator',
                'midi': 'separator', 'analysis': 'separator', 'package': 'separator', 'voice_metrics': '/opt/conda'}
# Existing stage callers also use these environment names for unrelated operations.
# Route these operations through the same resolver as preflight.
EXECUTION_BACKENDS = {'separate': 'separator', 'pitch': 'rmvpe', 'cover': 'yue',
                      'convert': 'seed_vc', 'convert_candidates': 'voice_metrics',
                      'evaluate_voice': 'voice_metrics', 'svc_metrics': 'voice_metrics'}
BASE_LOCKS = {'fish_diffusion': 'fish-diffusion-base.installed.txt', 'rvc': 'rvc-base.installed.txt', 'so_vits_svc': 'svc-base.installed.txt',
              'separator': 'pitch-base.installed.txt', 'rmvpe': 'pitch-base.installed.txt',
              'yue': 'yue-base.installed.txt', 'seed_vc': 'seed-vc-base.installed.txt',
              'voice_metrics': 'seed-vc-base.installed.txt'}
MODEL_GROUPS = {
    'sheetsage': ('generation_models.lock.json', ('SheetSage2', 'MERT-v2-FullSong')),
    'yue': ('generation_models.lock.json', ('YuE2-3B', 'YuE2-Vae')),
    'seed_vc': ('voice_models.lock.json', ('Seed-VC', 'BigVGAN', 'Whisper-small', 'RMVPE', 'CAMPPlus')),
    'rmvpe': ('voice_models.lock.json', ('RMVPE',)),
}


def python_for(root, backend):
    """Resolve one backend interpreter consistently for preflight and execution."""
    environment = Path(ENVIRONMENTS[backend])
    prefix = environment if environment.is_absolute() else Path(root) / 'envs' / environment
    return prefix / 'bin/python'


def execution_python(root, environment, operation):
    """Keep stage execution and preflight aligned without moving other backends."""
    if operation in EXECUTION_BACKENDS:
        return python_for(root, EXECUTION_BACKENDS[operation])
    prefix = Path(environment)
    if not prefix.is_absolute():
        prefix = Path(root) / 'envs' / prefix
    return prefix / 'bin/python'


def resource_data(root, backend):
    """Only the selected model entries; editing another model's lock does not invalidate this stage."""
    root = Path(root)
    if backend == 'fish_diffusion':
        return json.loads((root / 'configs/fish_diffusion_assets.lock.json').read_text())
    if backend == 'rvc':
        return json.loads((root / 'configs/rvc_assets.lock.json').read_text())
    if backend == 'so_vits_svc':
        return json.loads((root / 'configs/so_vits_svc_assets.lock.json').read_text())
    if backend == 'voice_metrics':
        data = json.loads((root / 'configs/voice_training_assets.lock.json').read_text())
        pitch = json.loads((root / 'configs/voice_models.lock.json').read_text())['RMVPE']
        return {'assets': [x for x in data['assets'] if x['scope'] == 'evaluation'], 'RMVPE': pitch}
    if backend in MODEL_GROUPS:
        name, keys = MODEL_GROUPS[backend]
        data = json.loads((root / 'configs' / name).read_text())
        return {key: data[key] for key in keys}
    if backend == 'separator':
        return json.loads((root / 'configs/separation_model.lock.json').read_text())
    if backend in {'review_stems', 'midi'}:
        data = json.loads((root / 'configs/review_models.lock.json').read_text())
        return {'assets': [a for a in data['assets'] if ('basic_pitch' in a['path']) == (backend == 'midi')]}
    return {}


def signature(root, backends):
    root = Path(root)
    result = {}
    for backend in sorted(set(backends)):
        env = ENVIRONMENTS[backend]
        freeze = root / 'configs' / BASE_LOCKS.get(backend, env + '.installed.txt')
        entry = {'models': resource_data(root, backend), 'environment': env,
                 'installed_lock_sha256': sha256(freeze)}
        if backend == 'seed_vc':
            entry['config_sha256'] = sha256(root / 'configs/seed_svc.local.yml')
        if backend == 'voice_metrics':
            entry['additional_requirements_sha256'] = sha256(root / 'configs/requirements-voice-adaptation.txt')
        # Small local model implementation files are executable dependencies, too.
        prefixes = {'fish_diffusion': ('third_party/fish-diffusion/fish_diffusion',), 'rvc': ('third_party/rvc/infer', 'third_party/rvc/tools', 'third_party/rvc/train'), 'so_vits_svc': ('third_party/so-vits-svc',), 'sheetsage': ('models/SheetSage2', 'models/MERT-v2-FullSong'),
                    'yue': ('models/YuE2-3B', 'models/YuE2-Vae', 'third_party/YuE'),
                    'seed_vc': ('third_party/seed-vc',), 'rmvpe': ('third_party/seed-vc/modules',),
                    'voice_metrics': ('third_party/seed-vc/modules',)}.get(backend, ())
        files = [p for prefix in prefixes for p in (root / prefix).rglob('*.py')
                 if '__pycache__' not in p.parts and not p.name.startswith('._')]
        entry['upstream_code'] = {str(p.relative_to(root)): sha256(p) for p in sorted(files)}
        result[backend] = entry
    return result


def requirements(root, backend):
    root = Path(root)
    required = [python_for(root, backend)]
    if backend == 'fish_diffusion':
        required += [root / a['path'] for a in resource_data(root, backend)['assets'] if a.get('scope') != 'training']
        required += [root / 'third_party/fish-diffusion/fish_diffusion/archs/diffsinger/diffsinger.py']
    if backend == 'rvc':
        required += [root / a['path'] for a in resource_data(root, backend)['assets'] if a.get('scope') != 'training']
        required += [root / 'third_party/rvc/infer/module/models.py']
    if backend == 'so_vits_svc':
        required += [root / a['path'] for a in resource_data(root, backend)['assets'] if a.get('scope') != 'training']
        required += [root / 'third_party/so-vits-svc/models.py']
    if backend == 'voice_metrics':
        required += [root / a['path'] for a in resource_data(root, backend)['assets']]
        required += [root / 'models/RMVPE/rmvpe.pt']
    if backend in MODEL_GROUPS:
        _, groups = MODEL_GROUPS[backend]
        names = {'SheetSage2': 'model.safetensors', 'MERT-v2-FullSong': 'model.safetensors',
                 'YuE2-3B': 'model.safetensors', 'YuE2-Vae': 'model.safetensors',
                 'Seed-VC': 'DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth',
                 'RMVPE': 'rmvpe.pt', 'CAMPPlus': 'campplus_cn_common.bin'}
        required += [root / 'models' / name / names[name] for name in groups if name in names]
        if backend == 'seed_vc':
            required += [root / 'third_party/seed-vc/inference.py', root / 'configs/seed_svc.local.yml']
    elif backend == 'separator':
        required += [root / 'models/separation/model_bs_roformer_ep_317_sdr_12.9755.ckpt']
    elif backend in {'review_stems', 'midi'}:
        required += [root / a['path'] for a in resource_data(root, backend)['assets']]
    return required


def preflight(root, backends=(), mp3=False):
    missing = [str(p) for b in sorted(set(backends)) for p in requirements(root, b) if not p.is_file()]
    if missing:
        raise FileNotFoundError('Selected modules need these files:\n' + '\n'.join(missing))
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg is not available; add /opt/conda/bin to PATH')
    if not shutil.which('ffprobe'):
        raise RuntimeError('ffprobe is not available; add the FFmpeg binary directory to PATH')
    if mp3:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], check=True, capture_output=True, text=True, timeout=30)
        if 'libmp3lame' not in result.stdout:
            raise RuntimeError('ffmpeg must support libmp3lame')
    for backend in sorted(set(backends)):
        command = [str(python_for(root, backend)), '-c',
                   'import importlib; ' + '; '.join('importlib.import_module(' + repr(name) + ')' for name in IMPORTS[backend])]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError(f'{backend} environment imports failed:\n{result.stderr[-2000:]}')
    if 'fish_diffusion' in backends:
        from ..fish import check_assets as check_fish_assets
        check_fish_assets(root)
    if 'so_vits_svc' in backends:
        from ..svc import check_assets
        check_assets(root)
    if 'rvc' in backends:
        from ..rvc import check_assets as check_rvc_assets
        check_rvc_assets(root)
    for b in set(backends) & {'midi', 'review_stems'}:
        for item in resource_data(root, b)['assets']:
            p = Path(root) / item['path']
            if p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']:
                raise RuntimeError(f'Model checksum mismatch: {p}')


IMPORTS = {
    'fish_diffusion': ['torch', 'torchaudio', 'pytorch_lightning', 'mmengine', 'parselmouth', 'loralib', 'wandb', 'transformers.models.hubert.modeling_hubert'],
    'rvc': ['torch', 'torchaudio', 'transformers.models.hubert.modeling_hubert', 'librosa', 'soundfile', 'faiss'],
    'so_vits_svc': ['torch', 'torchaudio', 'transformers.models.hubert.modeling_hubert', 'librosa', 'soundfile', 'faiss'],
    'sheetsage': ['torch', 'transformers'],
    'yue': ['yue2.pipeline', 'yue2.modeling_yue2', 'yue2.modeling_vae'],
    'separator': ['audio_separator.separator'],
    'seed_vc': ['torch', 'torchaudio', 'librosa', 'yaml', 'munch', 'dac.nn.quantize', 'transformers.models.whisper.modeling_whisper'],
    'rmvpe': ['torch', 'librosa'], 'review_stems': ['demucs.pretrained', 'audio_separator.separator'],
    'midi': ['basic_pitch.inference', 'onnxruntime', 'mido', 'pretty_midi'],
    'analysis': ['librosa'], 'package': ['mido', 'pretty_midi', 'soundfile'],
    'voice_metrics': ['speechbrain', 'librosa', 'soundfile'],
}
