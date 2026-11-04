"""Read-only checks for SVC in the existing base, including evaluation dependencies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .audio import sha256
from .common.paths import ROOT
from .common.resources import preflight
from .svc import check_assets


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--scope', choices=['all', 'training', 'inference'], default='all',
                   help='all also checks both source separation models')
    p.add_argument('--plan', action='store_true', help='All checks are read-only, even without this flag')
    return p


def check(a):
    if a.gpu < 0:
        raise ValueError('gpu must be nonnegative')
    if not Path(sys.executable).samefile('/opt/conda/bin/python'):
        raise ValueError('Run this check with /opt/conda/bin/python')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    if not torch.cuda.is_available():
        raise ValueError(f'CUDA unavailable for --gpu {a.gpu}')
    preflight(ROOT, ['so_vits_svc', 'voice_metrics'] + (['separator'] if a.scope == 'all' else []), mp3=True)
    lock = check_assets(ROOT, training=a.scope != 'inference')
    verified = [x['path'] for x in lock['assets'] if a.scope != 'inference' or x.get('scope') != 'training']
    for item in json.loads((ROOT / 'configs/voice_training_assets.lock.json').read_text())['assets']:
        if item['scope'] != 'evaluation':
            continue
        file = ROOT / item['path']
        if not file.is_file() or sha256(file) != item['sha256']:
            raise ValueError(f'Missing/corrupt evaluation asset: {file}')
        verified.append(item['path'])
    if a.scope == 'all':
        from .voice_data_separation import parser as separation_parser, preflight as separation_preflight
        settings = separation_preflight(separation_parser().parse_args(['--gpu', str(a.gpu)]))
        verified.extend('models/separation/' + name for name in settings['model_assets'])
    return {'status': 'passed', 'python': sys.executable, 'prefix': sys.prefix, 'gpu': a.gpu,
            'gpu_name': torch.cuda.get_device_name(0), 'scope': a.scope, 'writes': False,
            'verified_assets': [str(ROOT / p) for p in verified],
            'evaluation_f0': str(ROOT / 'models/RMVPE/rmvpe.pt'),
            'note': 'No installs, downloads, configuration creation, training or inference.'}


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    try:
        print(json.dumps(check(a), ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, subprocess.SubprocessError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
