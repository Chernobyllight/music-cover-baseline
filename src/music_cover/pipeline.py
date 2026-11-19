"""Public CLI; execution lives in engine.py and individual modules/."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import re
import sys
import uuid
from pathlib import Path
import yaml

from .engine import Pipeline
from .runtime import CUDA_DEFAULT_GPU, cuda_environment, memory_budget, resolve_device
from .common.io import write_json
from .common.lyrics import clean_lyrics
from .common.paths import ROOT, RunPaths
from .common.resources import preflight as check_resources
from .modules import load_modules, IDS, active_ids, normalize
from .voice import merge_voice_config

DEFAULT_REQUEST = {'name': 'cover', 'seed': 42, 'steps': 30, 'gpu': None, 'device': 'auto', 'memory_budget_gib': None, 'reference_start': 0.0,
                   'reference_duration': 15.0, 'transpose': '0', 'review': True}
REQUEST_KEYS = set(DEFAULT_REQUEST) | {'song', 'lyrics', 'reference', 'style'}


def apply_request_overrides(request, overrides, deferred=None):
    if deferred is None:
        deferred = {}
    current = request['device']
    target = overrides.get('device', current)
    for key in ('gpu', 'memory_budget_gib'):
        if key in overrides:
            deferred.pop(key, None)
        elif target != current and request.get(key) is not None:
            if target == 'auto':
                # An explicit auto choice can resolve back to the saved device.
                deferred[key] = current
            else:
                origin = deferred.pop(key, current)
                if origin != target and (origin, target) != ('auto', 'cuda'):
                    request[key] = None
    request.update(overrides)


def validate_device_options(request, selected):
    if request['device'] != 'mps':
        return
    if request.get('gpu') is not None:
        raise ValueError('--gpu cannot be combined with the selected MPS device')
    if any(IDS.index(stage) > IDS.index('06_mix') for stage in selected):
        raise ValueError('MPS supports stages 00–06. Use --until 06_mix.')


def read_config(path):
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict): raise ValueError(f'Configuration must be a mapping: {path}')
    return value


def module_config(root=ROOT):
    root = Path(root)
    pipeline = read_config(root / 'configs/pipeline.yaml')
    modules = {m.ID: dict(m.DEFAULTS) for m in load_modules()}
    for stage, path in pipeline.get('module_configs', {}).items():
        modules[normalize(stage)].update(read_config(root / path))
    return pipeline.get('request', {}), modules


def select_stages(request, *, stage=None, start=None, until=None):
    active = active_ids(request)
    if stage:
        result = [normalize(stage)]
    else:
        first = normalize(start) if start else active[0]
        last = normalize(until) if until else active[-1]
        if first not in active or last not in active: raise ValueError('Requested stage is disabled by --no-review')
        if active.index(first) > active.index(last): raise ValueError('--from must not follow --until')
        result = active[active.index(first):active.index(last) + 1]
    if any(s not in active for s in result): raise ValueError('Requested stage is disabled by --no-review')
    return result


def preflight(review=True):
    request = {**DEFAULT_REQUEST, 'review': review}
    modules = load_modules()
    check_resources(ROOT, [b for m in modules if m.ID in active_ids(request) for b in m.backends(request, m.DEFAULTS)], mp3=True)


def main():
    parser = argparse.ArgumentParser(description='分模块音乐改编：显式中间产物、单步执行与断点续跑')
    parser.add_argument('--demo', action='store_true')
    runs = parser.add_mutually_exclusive_group()
    runs.add_argument('--run', type=Path, help='指定新建或继续的模块化运行目录')
    runs.add_argument('--resume', type=Path, help='--run 的兼容别名；历史 v0.1 目录保持只读')
    parser.add_argument('--config', type=Path, help='任务 YAML/JSON，包含 request 与可选 modules 覆盖')
    parser.add_argument('--stage', help='只执行一个模块，要求上游产物完整且参数未失效')
    parser.add_argument('--from', dest='start', help='从该模块开始，校验并复用更早的依赖')
    parser.add_argument('--until', help='执行到该模块后停止')
    parser.add_argument('--force', action='store_true', help='选中的模块创建新 attempt；上游仍校验复用')
    parser.add_argument('--plan', action='store_true', help='只打印模块依赖和环境需求，不创建运行目录')
    parser.add_argument('--list-stages', action='store_true')
    parser.add_argument('--status', action='store_true', help='读取 --run 的状态')
    for name in ['song', 'lyrics', 'reference', 'style', 'name', 'gpu', 'transpose']:
        parser.add_argument('--' + name)
    for name in ['seed', 'steps']:
        parser.add_argument('--' + name, type=int)
    for name in ['reference-start', 'reference-duration']:
        parser.add_argument('--' + name, type=float)
    parser.add_argument('--review', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps'])
    parser.add_argument('--memory-budget-gib', type=float)
    parser.add_argument('--preflight', action='store_true', help='只检查选中模块的依赖')
    parser.add_argument('--voice-backend', choices=['seed_vc', 'so_vits_svc', 'rvc', 'fish_diffusion'])
    parser.add_argument('--voice-mode', choices=['zero_shot', 'finetuned'])
    parser.add_argument('--voice-profile', help='已导出的音色版本 voice.json，须与所选后端一致')
    parser.add_argument('--voice-candidates', type=int, help='微调推理候选数，默认 3')
    parser.add_argument('--voice-candidate-seeds', type=int, nargs='+', help='明确指定每个微调候选的随机种子')
    args = parser.parse_args()
    if args.list_stages:
        print('\n'.join(IDS))
        return
    if args.stage and (args.start or args.until): parser.error('--stage cannot be combined with --from/--until')
    if args.demo and args.resume: parser.error('--demo and --resume cannot be combined')
    run = args.run or args.resume
    if run: run = (ROOT / run.expanduser()).resolve()
    if args.status:
        if not run: parser.error('--status requires --run')
        print((run / 'stage_state.json').read_text())
        return
    try:
        default_request, configs = module_config()
        request = {**DEFAULT_REQUEST, **default_request}
        paths = RunPaths(ROOT, run)
        saved_device = None
        deferred = {}
        if args.demo:
            request.update(json.loads((ROOT / 'configs/demo.json').read_text()))
        if run and (run / 'request.json').exists():
            state = json.loads((run / 'stage_state.json').read_text())
            if state.get('schema_version') != 2:
                parser.error('历史 v0.1 运行目录保留不动。请指定新的 --run，或不传 --run 自动创建。')
            paths = RunPaths(ROOT, run, state)
            previous = json.loads((run / 'request.json').read_text())
            request.update(previous)
            saved_device = previous.get('device', 'cuda')
            request['device'] = saved_device
            saved = read_config(run / 'config.resolved.json')
            for stage, values in saved.get('modules', {}).items(): configs[stage].update(values)
        if args.config:
            path = (ROOT / args.config.expanduser()).resolve()
            task = read_config(path)
            unknown = set(task) - {'request', 'modules'}
            if unknown: parser.error(f'Unknown task sections: {sorted(unknown)}')
            apply_request_overrides(request, task.get('request', {}), deferred)
            for stage, values in task.get('modules', {}).items():
                stage = normalize(stage)
                if stage == '05_voice_conversion':
                    configs[stage] = merge_voice_config(configs[stage], values)
                else:
                    configs[stage].update(values)
        if set(request) - REQUEST_KEYS: parser.error(f'Unknown request keys: {sorted(set(request) - REQUEST_KEYS)}')
        voice_overrides = {key: value for key, value in {
            'backend': args.voice_backend, 'mode': args.voice_mode, 'voice_profile': args.voice_profile,
            'candidate_count': args.voice_candidates, 'candidate_seeds': args.voice_candidate_seeds,
        }.items() if value is not None}
        configs['05_voice_conversion'] = merge_voice_config(configs['05_voice_conversion'], voice_overrides)
        if configs['05_voice_conversion']['mode'] == 'zero_shot':
            configs['05_voice_conversion']['voice_profile'] = None
        elif configs['05_voice_conversion'].get('voice_profile'):
            configs['05_voice_conversion']['voice_profile'] = paths.project_relative(configs['05_voice_conversion']['voice_profile'])
        cli_overrides = {key: getattr(args, key) for key in REQUEST_KEYS if getattr(args, key, None) is not None}
        apply_request_overrides(request, cli_overrides, deferred)
        if request['device'] not in {'auto', 'mps', 'cuda'}:
            raise ValueError('device must be auto, cuda or mps')
        if request.get('memory_budget_gib') is not None:
            memory_budget(request['memory_budget_gib'], request['device'])
        request['transpose'] = str(request['transpose'])
        if request['transpose'] != 'auto':
            value = int(request['transpose'])
            if not -12 <= value <= 12: raise ValueError('transpose must be auto or -12..12')
            request['transpose'] = str(value)
        if not isinstance(request['review'], bool): raise ValueError('review must be true or false')
        if not math.isfinite(request['reference_start']) or not math.isfinite(request['reference_duration']) or request['reference_start'] < 0 or not 1 <= request['reference_duration'] <= 25:
            raise ValueError('reference start must be >= 0; duration must be between 1 and 25 seconds')
        if not isinstance(request['steps'], int) or isinstance(request['steps'], bool) or not 1 <= request['steps'] <= 200: raise ValueError('steps must be between 1 and 200')
        if not isinstance(request['seed'], int) or isinstance(request['seed'], bool) or not 0 <= request['seed'] < 2**32: raise ValueError('seed must be between 0 and 2**32 - 1')
        selected = select_stages(request, stage=args.stage, start=args.start, until=args.until)
        validate_device_options(request, selected)
        modules = {m.ID: m for m in load_modules()}
        for name, module in modules.items():
            unknown = set(configs[name]) - set(module.DEFAULTS)
            if unknown: raise ValueError(f'Unknown settings for {name}: {sorted(unknown)}')
            timeout = configs[name]['timeout_seconds']
            if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError('timeout_seconds must be positive and finite')
            module.params(request, configs[name])
        if args.plan:
            print(json.dumps({'selected': selected, 'modules': {name: {'depends_on': modules[name].requires(request),
                'backends': modules[name].backends(request, configs[name]), 'config': configs[name]} for name in selected},
                'note': 'This is a static execution plan. Cache checks happen when running.'}, ensure_ascii=False, indent=2))
            return
        # CPU-only stages and static commands need no accelerator discovery.
        selected_backends = {b for name in selected for b in modules[name].backends(request, configs[name])}
        if selected_backends - {'analysis', 'package'}:
            gpu = request['gpu'] if deferred.get('gpu', 'cuda') == 'cuda' else None
            request['device'] = resolve_device(
                request['device'], gpu=gpu if gpu is not None else CUDA_DEFAULT_GPU)
        elif request['device'] == 'auto' and saved_device:
            request['device'] = saved_device
        if request['device'] != 'auto':
            for key, origin in deferred.items():
                if origin != request['device']:
                    request[key] = None
        validate_device_options(request, selected)
        if request['device'] == 'cuda' and request.get('gpu') is None:
            request['gpu'] = CUDA_DEFAULT_GPU
        if request['device'] != 'auto':
            request['memory_budget_gib'] = memory_budget(request.get('memory_budget_gib'), request['device'])
        if request['device'] == 'mps':
            from .common.resources import environment_for
            for backend in sorted(selected_backends):
                environment_for(backend, 'mps')
        if args.preflight:
            if '05_voice_conversion' in selected:
                from .voice import resolve_voice
                c = configs['05_voice_conversion']
                resolve_voice(c['mode'], c['voice_profile'], backend=c['backend'])
            environment = cuda_environment(request['gpu']) if request['device'] == 'cuda' else None
            check_resources(ROOT, selected_backends, mp3='06_mix' in selected,
                            device=request['device'], environment=environment)
            print('Selected module preflight OK')
            return
        for key in ['song', 'lyrics', 'reference', 'style']:
            if not request.get(key): parser.error(f'--{key} is required (or use --config / --demo)')
        for key in ['song', 'lyrics', 'reference']:
            request[key] = str(paths.resolve(request[key]))
            if not Path(request[key]).is_file(): parser.error(f'File does not exist: {request[key]}')
        if run is None:
            name = re.sub(r'[^\w-]+', '_', request['name'])[:60]
            run = ROOT / 'runs' / (dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + name + '_' + uuid.uuid4().hex[:6])
        print(f'RUN: {run}', flush=True)
        with Pipeline(run, request, configs) as pipeline:
            pipeline.run_all(selected, force=args.force)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
