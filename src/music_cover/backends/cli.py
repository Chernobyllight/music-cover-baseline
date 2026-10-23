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
    'convert': ('seed_vc', 'convert', False),
    'pitch': ('rmvpe', 'pitch', False),
    'review_stems': ('review_stems', 'separate_review', True),
    'review_midi': ('midi', 'transcribe_stems', True),
    'review_package': ('package', 'build_review_package', True),
    'evaluate': ('evaluation', 'evaluate', True),
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
