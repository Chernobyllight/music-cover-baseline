"""Artifact-driven stage execution, scoped caching and atomic publication."""
from __future__ import annotations
import contextlib
import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .audio import sha256, inspect_audio
from .common.io import identity, write_json, artifact_paths
from .common.paths import ROOT
from .common import resources
from .modules import load_modules, active_ids

SCHEMA_VERSION = 2
CONTRACT_VERSION = 1


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


class DependencyError(RuntimeError):
    pass


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()


class Pipeline:
    def __init__(self, run, request, config=None, *, root=ROOT, modules=None, invocation=None):
        self.root, self.run, self.request = Path(root).resolve(), Path(run).resolve(), request
        self.modules = {m.ID: m for m in (modules if modules is not None else load_modules())}
        self.config = {key: {**m.DEFAULTS, **(config or {}).get(key, {})} for key, m in self.modules.items()}
        for key, module in self.modules.items():
            unknown = set(self.config[key]) - set(module.DEFAULTS)
            if unknown: raise ValueError(f'Unknown settings for {key}: {sorted(unknown)}')
            if self.config[key].get('timeout_seconds', 10800) <= 0: raise ValueError('timeout_seconds must be positive')
            module.params(request, self.config[key])
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run / '.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state_path = self.run / 'stage_state.json'
            self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {'schema_version': SCHEMA_VERSION, 'stages': {}}
            if self.state.get('schema_version') != SCHEMA_VERSION:
                raise ValueError('This is a legacy v0.1 run. Its files are preserved; choose a new --run directory for modular execution.')
        except BaseException:
            self.lock.close()
            raise
        self.invocation = invocation or sys.argv
        self.resolved = {}
        self.events = []
        self.runtime_cache = {}
        self.environment = os.environ.copy()
        self.environment.update({'CUDA_VISIBLE_DEVICES': str(request.get('gpu', '0')), 'OMP_NUM_THREADS': '4',
            'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'HF_HOME': str(self.root / 'cache/huggingface'),
            'HF_HUB_DOWNLOAD_TIMEOUT': '180', 'HF_HUB_ETAG_TIMEOUT': '60', 'TOKENIZERS_PARALLELISM': 'false',
            'PYTHONUNBUFFERED': '1', 'USE_TF': '0', 'USE_FLAX': '0', 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1',
            'PYTHONPATH': str(self.root / 'src') + os.pathsep + self.environment.get('PYTHONPATH', '')})
        self.current = None

    def close(self): self.lock.close()
    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def output(self, stage, role):
        return self.resolved[stage][role]

    def all_outputs(self, stage, prefix=''):
        return {prefix + key: path for key, path in self.resolved[stage].items()}

    def runtime(self, module):
        backends = tuple(sorted(module.backends(self.request, self.config[module.ID])))
        if backends not in self.runtime_cache:
            self.runtime_cache[backends] = resources.signature(self.root, backends)
        return self.runtime_cache[backends]

    def fingerprint(self, module, inputs, params):
        files = [Path(module.__file__), self.root / 'src/music_cover/common/io.py']
        files += [self.root / 'src/music_cover' / name for name in getattr(module, 'CODE', ())]
        # The score parser is shared by pitch adaptation and YuE's score duration check.
        if 'pitch.py' in getattr(module, 'CODE', ()):
            helper = self.root / 'third_party/YuE/skills/yue2-music/scripts/abc_tools.py'
            if helper.is_file(): files.append(helper)
        code = {str(p.relative_to(self.root)) if p.is_relative_to(self.root) else str(p): sha256(p) for p in files}
        runtime = dict(self.runtime(module))
        control_lock = self.root / 'configs/control.installed.txt'
        runtime['control_installed_sha256'] = sha256(control_lock) if control_lock.is_file() else None
        input_records = {}
        for role, path in inputs.items():
            path = Path(path).resolve()
            if not path.is_file(): raise FileNotFoundError(f'Missing input {role}: {path}')
            item = {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}
            for stage, record in self.state['stages'].items():
                for producer_role, output in record.get('outputs', {}).items():
                    if output['path'] == str(path):
                        item.update(producer_stage=stage, producer_role=producer_role, producer_attempt=record['attempt'],
                                    timeline=output.get('timeline'), contract_version=record.get('contract_version'))
                        break
            input_records[role] = item
        value = {'contract_version': getattr(module, 'CONTRACT_VERSION', CONTRACT_VERSION), 'stage': module.ID,
                 'inputs': {key: value['sha256'] for key, value in input_records.items()},
                 'params': params, 'code': code, 'runtime': runtime}
        return identity(value), input_records, value

    def cached(self, name, fingerprint):
        record = self.state['stages'].get(name, {})
        if record.get('fingerprint') != fingerprint or record.get('status') != 'succeeded': return None
        try:
            marker = Path(record['attempt']) / 'outputs.json'
            if sha256(marker) != record['outputs_manifest_sha256']: return None
            manifest = json.loads(marker.read_text())
            if manifest['fingerprint'] != fingerprint or manifest['status'] != 'succeeded': return None
            outputs = record['outputs']
            if not outputs: return None
            for item in outputs.values():
                path = Path(item['path'])
                if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']: return None
            return {key: Path(value['path']) for key, value in outputs.items()}
        except (OSError, ValueError, KeyError):
            return None

    def execute(self, env, operation, output, *arguments):
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        command = [str(self.root / 'envs' / env / 'bin/python'), '-m', 'music_cover.backends.cli', operation,
                   '--output', str(output), *map(str, arguments)]
        attempt, log_path, commands, params = self.current
        commands['backend_commands'].append({'argv': command, 'cwd': str(self.root), 'environment': env,
                                             'gpu': self.request.get('gpu', '0'), 'started_at': now()})
        write_json(attempt / 'command.json', commands)
        print(f'[backend] {operation} ({env}); log: {log_path}', flush=True)
        with log_path.open('a') as log:
            process = subprocess.run(command, cwd=self.root, env=self.environment, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=params.get('timeout_seconds', 10800))
        commands['backend_commands'][-1].update(returncode=process.returncode, finished_at=now())
        write_json(attempt / 'command.json', commands)
        if process.returncode:
            tail = log_path.read_text(errors='replace')[-4500:]
            raise RuntimeError(f'{operation} failed ({process.returncode}); {log_path}\n{tail}')

    def process(self, name, selected, force=False):
        if name in self.resolved: return self.resolved[name]
        module = self.modules[name]
        for dependency in module.requires(self.request): self.process(dependency, selected, force)
        inputs = module.inputs(self)
        params = module.params(self.request, self.config[name])
        fingerprint, input_records, provenance = self.fingerprint(module, inputs, params)
        cached = self.cached(name, fingerprint)
        if cached and not (force and name in selected):
            self.resolved[name] = cached
            self.events.append({'stage': name, 'action': 'reused', 'attempt': self.state['stages'][name]['attempt']})
            print(f'[reuse] {name}', flush=True)
            return cached
        if name not in selected:
            raise DependencyError(f'{name} is missing, changed or corrupt. Run it first, or use --from {name}. No later stage was allowed to consume it.')
        resources.preflight(self.root, module.backends(self.request, self.config[name]), mp3=name == '06_mix')
        base = self.run / 'stages' / name
        base.mkdir(parents=True, exist_ok=True)
        numbers = [int(p.name[8:]) for p in base.glob('attempt_*') if p.name[8:].isdigit()]
        attempt = base / f'attempt_{max(numbers, default=0) + 1:03d}'
        output = attempt / 'artifacts'
        output.mkdir(parents=True)
        record = {'status': 'running', 'stage': name, 'contract_version': getattr(module, 'CONTRACT_VERSION', CONTRACT_VERSION),
                  'fingerprint': fingerprint, 'attempt': str(attempt), 'params': params, 'started_at': now()}
        self.state['stages'][name] = record
        write_json(self.state_path, self.state)
        write_json(attempt / 'inputs.json', {'contract_version': getattr(module, 'CONTRACT_VERSION', CONTRACT_VERSION), 'artifacts': input_records})
        write_json(attempt / 'config.json', {'stage': name, 'parameters': params, 'provenance': provenance})
        commands = {'invocation': self.invocation, 'backend_commands': [], 'cwd': str(self.root)}
        write_json(attempt / 'command.json', commands)
        log_path = attempt / 'process.log'
        log_path.touch()
        self.current = (attempt, log_path, commands, params)
        start = time.monotonic()
        print(f'[run] {name} -> {attempt}', flush=True)
        try:
            with log_path.open('a') as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                paths = module.run(self, output, inputs, params)
            if not paths: raise RuntimeError(f'{name} returned no artifacts')
            # Everything emitted by the backend is tracked, not just the primary WAV.
            known = {Path(p).resolve() for p in paths.values()}
            paths.update({'file:' + key: path for key, path in artifact_paths(output).items() if path.resolve() not in known})
            records, portable = {}, {}
            for role, path in paths.items():
                path = Path(path).resolve()
                if not path.is_relative_to(output.resolve()): raise ValueError(f'Output outside attempt: {path}')
                if not path.is_file() or not path.stat().st_size: raise RuntimeError(f'Missing stage artifact: {path}')
                metadata = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path),
                            'media_kind': path.suffix.lstrip('.'), 'timeline': getattr(module, 'TIMELINE', {}).get(role, getattr(module, 'TIMELINE', {}).get('*'))}
                if path.suffix.lower() in {'.wav', '.mp3', '.flac'}:
                    metadata['audio'] = inspect_audio(path)
                    metadata['time_origin_seconds'] = 0.0
                elif path.suffix == '.json':
                    json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f'Non-finite JSON: {path}')))
                records[role] = metadata
                portable[role] = {**metadata, 'path': str(path.relative_to(attempt))}
            write_json(attempt / 'checks.json', {'status': 'passed', 'files_checked': len(records),
                        'checks': ['nonempty', 'sha256', 'json_finite', 'audio_finite'],
                        'subjective_quality': 'requires_human_review'})
            write_json(attempt / 'outputs.json', {'contract_version': getattr(module, 'CONTRACT_VERSION', CONTRACT_VERSION), 'stage': name,
                        'status': 'succeeded', 'fingerprint': fingerprint, 'artifacts': portable})
            record.update(status='succeeded', outputs=records, wall_seconds=time.monotonic() - start, finished_at=now(),
                          outputs_manifest_sha256=sha256(attempt / 'outputs.json'))
            write_json(attempt / 'result.json', record)
            self.resolved[name] = {key: Path(item['path']) for key, item in records.items()}
            self.events.append({'stage': name, 'action': 'executed', 'attempt': str(attempt), 'wall_seconds': record['wall_seconds']})
        except BaseException as error:
            record.update(status='failed', error=str(error), wall_seconds=time.monotonic() - start, finished_at=now())
            write_json(attempt / 'failure.json', record)
            write_json(attempt / 'checks.json', {'status': 'failed', 'error': str(error)})
            raise
        finally:
            self.current = None
            write_json(self.state_path, self.state)
        print(f"[done] {name}: {record['wall_seconds']:.1f}s", flush=True)
        return self.resolved[name]

    def publish(self):
        artifact_dir = Path(self.state['stages']['11_delivery']['attempt']) / 'artifacts'
        destination = self.run / 'deliverables'
        # A complete previous delivery stays available until a validated replacement exists.
        if destination.exists() and not destination.is_symlink():
            history = self.run / 'deliverables_history'
            history.mkdir(exist_ok=True)
            destination.rename(history / (dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]))
        temporary = self.run / ('.delivery_' + uuid.uuid4().hex)
        temporary.symlink_to(artifact_dir.relative_to(self.run), target_is_directory=True)
        temporary.replace(destination)
        reviews = self.run / 'reviews'
        reviews.mkdir(exist_ok=True)
        template = destination / 'review/review_scores.csv'
        if template.is_file() and not (reviews / 'review_scores.csv').exists():
            text = template.read_text(encoding='utf-8-sig').replace('../final.wav', '../deliverables/final.wav')
            text = text.replace(',audio/', ',../deliverables/review/audio/').replace(',midi/', ',../deliverables/review/midi/')
            (reviews / 'review_scores.csv').write_text(text, encoding='utf-8-sig')
        if not (reviews / 'README.txt').exists():
            (reviews / 'README.txt').write_text('请在此填写人工评审；各 attempt/artifacts 保持不变。每轮评审请另存带评审人及日期的文件。\n')

    def run_all(self, selected=None, force=False):
        selected = list(selected) if selected is not None else [s for s in active_ids(self.request) if s in self.modules]
        self.state.pop('error', None)
        self.state['active_stages'] = active_ids(self.request)
        self.state.update(status='running', request=self.request, selected_stages=selected, delivery_status='pending', updated_at=now())
        write_json(self.run / 'request.json', self.request)
        write_json(self.run / 'config.resolved.json', {'request': self.request, 'modules': self.config})
        write_json(self.state_path, self.state)
        try:
            for name in selected: self.process(name, set(selected), force)
            if '11_delivery' in selected:
                self.publish()
                self.state.update(status='completed', delivery_status='current', subjective_quality='requires_listening')
            else:
                self.state['status'] = 'partial'
            self.state['last_execution'] = self.events
        except BaseException as error:
            self.state.update(status='failed', error=str(error), last_execution=self.events)
            raise
        finally:
            self.state['updated_at'] = now()
            write_json(self.state_path, self.state)
            write_json(self.run / 'manifest.json', self.state)
            write_json(self.run / 'executions' / (dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6] + '.json'),
                       {'status': self.state['status'], 'request': self.request, 'selected': selected, 'events': self.events,
                        'error': self.state.get('error'), 'invocation': self.invocation})
        print(f"RESULTS ({self.state['status']}): {self.run}", flush=True)
