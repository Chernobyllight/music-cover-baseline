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
from .common.io import write_json
from .common.lyrics import clean_lyrics
from .common.paths import ROOT
from .common.resources import preflight as check_resources
from .modules import load_modules, IDS, active_ids, normalize

DEFAULT_REQUEST = {'name': 'cover', 'seed': 42, 'steps': 30, 'gpu': '0', 'reference_start': 0.0,
                   'reference_duration': 15.0, 'transpose': '0', 'review': True}
REQUEST_KEYS = set(DEFAULT_REQUEST) | {'song', 'lyrics', 'reference', 'style'}


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
    parser.add_argument('--preflight', action='store_true', help='只检查选中模块的依赖')
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
        if args.demo:
            request.update(json.loads((ROOT / 'configs/demo.json').read_text()))
        if run and (run / 'request.json').exists():
            state = json.loads((run / 'stage_state.json').read_text())
            if state.get('schema_version') != 2:
                parser.error('历史 v0.1 运行目录保留不动。请指定新的 --run，或不传 --run 自动创建。')
            request.update(json.loads((run / 'request.json').read_text()))
            saved = read_config(run / 'config.resolved.json')
            configs.update(saved.get('modules', {}))
        if args.config:
            path = (ROOT / args.config.expanduser()).resolve()
            task = read_config(path)
            unknown = set(task) - {'request', 'modules'}
            if unknown: parser.error(f'Unknown task sections: {sorted(unknown)}')
            request.update(task.get('request', {}))
            for stage, values in task.get('modules', {}).items(): configs[normalize(stage)].update(values)
        if set(request) - REQUEST_KEYS: parser.error(f'Unknown request keys: {sorted(set(request) - REQUEST_KEYS)}')
        for key in REQUEST_KEYS:
            value = getattr(args, key, None)
            if value is not None: request[key] = value
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
        if args.preflight:
            check_resources(ROOT, [b for name in selected for b in modules[name].backends(request, configs[name])], mp3='06_mix' in selected)
            print('Selected module preflight OK')
            return
        for key in ['song', 'lyrics', 'reference', 'style']:
            if not request.get(key): parser.error(f'--{key} is required (or use --config / --demo)')
        for key in ['song', 'lyrics', 'reference']:
            path = Path(request[key]).expanduser()
            request[key] = str((ROOT / path).resolve())
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
