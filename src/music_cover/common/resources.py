"""Backend-scoped model/runtime identities and preflight requirements."""
import json
import shutil
import subprocess
from pathlib import Path
from ..audio import sha256

ENVIRONMENTS = {'sheetsage': 'sheetsage2', 'yue': 'yue2', 'separator': 'separator',
                'seed_vc': 'seed-vc', 'rmvpe': 'seed-vc', 'review_stems': 'separator',
                'midi': 'separator', 'analysis': 'separator', 'package': 'separator'}
MODEL_GROUPS = {
    'sheetsage': ('generation_models.lock.json', ('SheetSage2', 'MERT-v2-FullSong')),
    'yue': ('generation_models.lock.json', ('YuE2-3B', 'YuE2-Vae')),
    'seed_vc': ('voice_models.lock.json', ('Seed-VC', 'BigVGAN', 'Whisper-small', 'RMVPE', 'CAMPPlus')),
    'rmvpe': ('voice_models.lock.json', ('RMVPE',)),
}


def resource_data(root, backend):
    """Only the selected model entries; editing another model's lock does not invalidate this stage."""
    root = Path(root)
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
        freeze = root / 'configs' / (env + '.installed.txt')
        entry = {'models': resource_data(root, backend), 'environment': env,
                 'installed_lock_sha256': sha256(freeze)}
        if backend == 'seed_vc':
            entry['config_sha256'] = sha256(root / 'configs/seed_svc.local.yml')
        # Small local model implementation files are executable dependencies, too.
        prefixes = {'sheetsage': ('models/SheetSage2', 'models/MERT-v2-FullSong'),
                    'yue': ('models/YuE2-3B', 'models/YuE2-Vae', 'third_party/YuE'),
                    'seed_vc': ('third_party/seed-vc',), 'rmvpe': ('third_party/seed-vc/modules',)}.get(backend, ())
        files = [p for prefix in prefixes for p in (root / prefix).rglob('*.py') if '__pycache__' not in p.parts]
        entry['upstream_code'] = {str(p.relative_to(root)): sha256(p) for p in sorted(files)}
        result[backend] = entry
    return result


def requirements(root, backend):
    root = Path(root)
    required = [root / 'envs' / ENVIRONMENTS[backend] / 'bin/python']
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
    if mp3:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], check=True, capture_output=True, text=True, timeout=30)
        if 'libmp3lame' not in result.stdout:
            raise RuntimeError('ffmpeg must support libmp3lame')
    for backend in sorted(set(backends)):
        command = [str(Path(root) / 'envs' / ENVIRONMENTS[backend] / 'bin/python'), '-c',
                   'import importlib; ' + '; '.join('importlib.import_module(' + repr(name) + ')' for name in IMPORTS[backend])]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError(f'{backend} environment imports failed:\n{result.stderr[-2000:]}')
    for b in set(backends) & {'midi', 'review_stems'}:
        for item in resource_data(root, b)['assets']:
            p = Path(root) / item['path']
            if p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']:
                raise RuntimeError(f'Model checksum mismatch: {p}')


IMPORTS = {
    'sheetsage': ['torch', 'transformers'], 'yue': ['yue2'],
    'separator': ['audio_separator.separator'], 'seed_vc': ['torch', 'librosa', 'yaml'],
    'rmvpe': ['torch', 'librosa'], 'review_stems': ['demucs.pretrained', 'audio_separator.separator'],
    'midi': ['basic_pitch.inference', 'onnxruntime', 'mido', 'pretty_midi'],
    'analysis': ['librosa'], 'package': ['mido', 'pretty_midi', 'soundfile'],
}
