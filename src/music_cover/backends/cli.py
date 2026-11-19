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
    'convert_fish': ('fish_diffusion', 'convert', True),
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
    parser.add_argument('stage', choices=[*OPERATIONS, "probe"])
    for name in ['input', 'request', 'score', 'reference']:
        parser.add_argument('--' + name)
    parser.add_argument('--output')
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps'])
    parser.add_argument('--memory-budget-gib', type=float)
    parser.add_argument('--environment', choices=['control', 'sheetsage2', 'yue2', 'separator', 'seed-vc'])
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--voice-mode', choices=['zero_shot', 'finetuned'], default='zero_shot')
    parser.add_argument('--voice-profile')
    parser.add_argument('--checkpoint')
    parser.add_argument('--model-config')
    args = parser.parse_args()
    from ..runtime import ForwardAudit, MemoryMonitor, memory_budget, probe, resolve_device
    if args.stage == 'probe':
        if args.memory_budget_gib is not None:
            parser.error('--memory-budget-gib is only supported for device-aware inference')
        if args.environment == 'seed-vc':
            import sys
            from ..common.paths import ROOT
            sys.path.insert(0, str(ROOT / 'third_party/seed-vc'))
        print(json.dumps(probe(args.device or 'cuda', args.environment)))
        return
    if args.environment is not None:
        parser.error('--environment is only supported for probe')
    if not args.output:
        parser.error('--output is required for inference')
    module, function, request_api = OPERATIONS[args.stage]
    if request_api and (args.device is not None or args.memory_budget_gib is not None):
        parser.error(f'{args.stage} does not accept --device or --memory-budget-gib')
    if args.stage == 'transcribe_final' and args.device == 'mps':
        parser.error('transcribe_final is outside the supported MPS stages 00–06')
    Path(args.output).mkdir(parents=True, exist_ok=True)
    function = getattr(importlib.import_module('music_cover.backends.' + module), function)
    start = time.monotonic()
    if request_api:
        function(json.loads(Path(args.request).read_text()), Path(args.output))
    else:
        args.device = resolve_device(args.device or 'cuda')
        if args.stage == 'transcribe_final' and args.device == 'mps':
            parser.error('transcribe_final is outside the supported MPS stages 00–06')
        if args.device == 'cuda' and args.stage != 'cover' and args.memory_budget_gib is not None:
            parser.error(f'{args.stage} does not support --memory-budget-gib on CUDA')
        args.memory_budget_gib = memory_budget(args.memory_budget_gib, args.device)
        if args.device == 'mps':
            monitor = MemoryMonitor(args.device, args.memory_budget_gib)
            audit = ForwardAudit(args.device)
            try:
                with monitor, audit:
                    function(args)
                if not audit.layers:
                    raise RuntimeError('No learned-layer MPS forward observed')
            finally:
                from ..common.io import write_json
                write_json(Path(args.output) / 'runtime.json', {**monitor.report(), 'forward_layers': audit.layers})
        else:
            function(args)
    print(json.dumps({'stage': args.stage, 'wall_seconds': time.monotonic() - start}), flush=True)


if __name__ == '__main__':
    main()
