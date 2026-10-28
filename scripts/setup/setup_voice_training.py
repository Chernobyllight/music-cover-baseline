"""Download only explicitly requested training/evaluation assets, pinned and hashed."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scope', choices=['training', 'evaluation', 'all'], default='training')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    for row in json.loads((ROOT / 'configs/voice_training_assets.lock.json').read_text())['assets']:
        if args.scope != 'all' and row['scope'] != args.scope: continue
        target = ROOT / row['path']
        if not target.is_file() or digest(target) != row['sha256']:
            if args.check_only: raise RuntimeError(f'Missing/corrupt asset: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + '.download')
            url = f"https://huggingface.co/{row['repo']}/resolve/{row['revision']}/{row['file']}"
            subprocess.run(['curl', '-fL', '--retry', '2', '--connect-timeout', '20', '--max-time', '600', url, '-o', str(temp)], check=True)
            if digest(temp) != row['sha256']: raise RuntimeError(f'Download checksum mismatch: {temp}')
            temp.replace(target)
        print(f"OK {row['path']}")


if __name__ == '__main__': main()
