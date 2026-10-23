#!/usr/bin/env python3
"""Verify final payloads and both legacy/modular run manifests without model inference."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from music_cover.common.validation import verify_delivery
from music_cover.audio import sha256


def verify(run):
    run = Path(run).resolve()
    state = json.loads((run / 'manifest.json').read_text())
    if state['status'] != 'completed': raise ValueError('Run has not completed')
    result = verify_delivery(run / 'deliverables', state['request'].get('review', True))
    if state.get('schema_version') == 2:
        for name in state['active_stages']:
            record = state['stages'][name]
            if record['status'] != 'succeeded': raise ValueError(f'Incomplete module: {name}')
            marker = Path(record['attempt']) / 'outputs.json'
            if sha256(marker) != record['outputs_manifest_sha256']: raise ValueError(f'Manifest checksum mismatch: {name}')
            for item in record['outputs'].values():
                if sha256(item['path']) != item['sha256']: raise ValueError(f'Artifact checksum mismatch: {item["path"]}')
        result['checked_stages'] = len(state['active_stages'])
    elif state['request'].get('review', True):
        for name in ['07_review_stems','08_final_transcribe','09_review_midi','10_review_package']:
            if state['stages'][name]['status'] != 'succeeded': raise ValueError(f'Incomplete legacy stage: {name}')
    result['request'] = state['request']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__': main()
