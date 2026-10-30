"""One isolated process per backend; no heavyweight imports in the controller."""
import argparse
import importlib
import json
import time
from pathlib import Path

OPERATIONS = {
    'transcribe': ('sheetsage', 'transcribe', False),
    'transcribe_final': ('sheetsage', 'transcribe', False),
    'cover': ('yue', 'cover', False),
    'separate': ('separator', 'separate', False),
    'svc_metrics': ('svc_metrics', 'evaluate', True),
    'convert_svc': ('so_vits_svc', 'convert', True),
    'convert_rvc': ('rvc', 'convert', True),
    'convert': ('seed_vc', 'convert', False),
    'convert_candidates': ('voice_candidates', 'generate', True),
    'pitch': ('rmvpe', 'pitch', False),
    'review_stems': ('review_stems', 'separate_review', True),
    'review_midi': ('midi', 'transcribe_stems', True),
    'review_package': ('package', 'build_review_package', True),
    'evaluate': ('evaluation', 'evaluate', True),
    'evaluate_voice': ('voice_metrics', 'evaluate_voice', True),
    'validate_delivery': ('evaluation', 'validate_delivery', True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=OPERATIONS)
    for name in ['input', 'request', 'score', 'reference']:
        parser.add_argument('--' + name)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--voice-mode', choices=['zero_shot', 'finetuned'], default='zero_shot')
    parser.add_argument('--voice-profile')
    parser.add_argument('--checkpoint')
    parser.add_argument('--model-config')
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    module, function, request_api = OPERATIONS[args.stage]
    function = getattr(importlib.import_module('music_cover.backends.' + module), function)
    start = time.monotonic()
    if request_api:
        function(json.loads(Path(args.request).read_text()), Path(args.output))
    else:
        function(args)
    print(json.dumps({'stage': args.stage, 'wall_seconds': time.monotonic() - start}), flush=True)


if __name__ == '__main__':
    main()
