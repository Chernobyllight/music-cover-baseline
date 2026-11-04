"""Project-relative inputs and run-relative artifacts, including legacy records."""
from copy import deepcopy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class RunPaths:
    def __init__(self, root, run, state=None):
        self.root = Path(root).resolve()
        self.run = Path(run).resolve() if run is not None else None
        self.state = state or {}
        self.legacy_roots = set()
        if self.run is None:
            return
        # Shared-upstream runs can refer to a sibling run in the same project.
        # Derive old roots from the recorded suffix, not a mount-name list.
        if self.run.parent.is_relative_to(self.root):
            for stage, record in self.state.get('stages', {}).items():
                old = Path(record.get('attempt', '.'))
                if old.is_absolute() and old.parts[-3:-1] == ('stages', stage):
                    suffix = self.run.parent.relative_to(self.root) / old.parents[2].name / 'stages' / stage / old.name
                    if old.parts[-len(suffix.parts):] == suffix.parts:
                        self.legacy_roots.add(Path(*old.parts[:-len(suffix.parts)]))
        for stage, record in self.state.get('stages', {}).items():
            raw = record.get('attempt')
            if not raw:
                continue
            local = self.attempt(stage, raw)
            # Execution records identify the old checkout without assuming any
            # server, mount name or project directory name.
            command = local / 'command.json'
            if command.is_file():
                try:
                    cwd = json.loads(command.read_text()).get('cwd')
                    if cwd and Path(cwd).is_absolute():
                        self.legacy_roots.add(Path(cwd))
                except (OSError, ValueError):
                    pass
            old = Path(raw)
            if old.is_absolute() and self.run.is_relative_to(self.root):
                suffix = self.run.relative_to(self.root) / 'stages' / stage / old.name
                if old.parts[-len(suffix.parts):] == suffix.parts:
                    self.legacy_roots.add(Path(*old.parts[:-len(suffix.parts)]))

    def resolve(self, value):
        """Resolve a project input; never search by basename or remap external data."""
        path = Path(value).expanduser()
        if not path.is_absolute():
            return (self.root / path).resolve()
        if path.is_relative_to(self.root):
            return path.resolve()
        for old in sorted(self.legacy_roots, key=lambda p: len(p.parts), reverse=True):
            if path.is_relative_to(old):
                return (self.root / path.relative_to(old)).resolve()
        return path.resolve()

    def project_relative(self, value):
        path = self.resolve(value)
        return str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)

    def request(self, value, *, portable=False):
        result = deepcopy(value)
        for key in ('song', 'lyrics', 'reference'):
            if result.get(key):
                result[key] = self.project_relative(result[key]) if portable else str(self.resolve(result[key]))
        return result

    def run_path(self, value):
        path = Path(value)
        if path.is_absolute():
            return self.resolve(path)
        result = (self.run / path).resolve()
        if not (result.is_relative_to(self.run) or result.is_relative_to(self.root)):
            raise ValueError(f'Run artifact escapes its project: {value}')
        return result

    def run_relative(self, value):
        path = Path(value).resolve()
        return os.path.relpath(path, self.run) if (path.is_relative_to(self.root) or path.is_relative_to(self.run)) else str(path)

    def attempt(self, stage, value):
        old = Path(value)
        if self.state.get('path_base') == 'run':
            return self.run_path(old)
        mapped = self.resolve(old)
        if mapped.is_dir():
            return mapped
        # A legacy attempt is anchored to its stage, even after a run is renamed.
        if (Path(stage).name == stage and old.name.startswith('attempt_')
                and old.name[8:].isdigit() and old.parts[-3:-1] == ('stages', stage)):
            local = self.run / 'stages' / stage / old.name
            if local.is_dir():
                return local
        return mapped

    def restore_state(self):
        result = deepcopy(self.state)
        for stage, record in result.get('stages', {}).items():
            if not record.get('attempt'):
                continue
            old_attempt = Path(record['attempt'])
            local = self.attempt(stage, old_attempt)
            record['attempt'] = str(local)
            for item in record.get('outputs', {}).values():
                old = Path(item['path'])
                if self.state.get('path_base') == 'run':
                    new = self.run_path(old)
                elif old.is_absolute() and old.is_relative_to(old_attempt):
                    new = (local / old.relative_to(old_attempt)).resolve()
                else:
                    new = self.resolve(old)
                item['path'] = str(new)
        if 'request' in result:
            result['request'] = self.request(result['request'])
        for event in result.get('last_execution', []):
            if event.get('attempt') and event.get('stage'):
                event['attempt'] = str(self.attempt(event['stage'], event['attempt']))
        result.pop('path_base', None)
        return result

    def portable_state(self, state):
        result = deepcopy(state)
        result['path_base'] = 'run'
        if 'request' in result:
            result['request'] = self.request(result['request'], portable=True)
        for record in result.get('stages', {}).values():
            if record.get('attempt'):
                record['attempt'] = self.run_relative(record['attempt'])
            for item in record.get('outputs', {}).values():
                item['path'] = self.run_relative(item['path'])
        for event in result.get('last_execution', []):
            if event.get('attempt'):
                event['attempt'] = self.run_relative(event['attempt'])
        return result


def read_run_state(run, root=ROOT):
    """Use this reader when consuming main-pipeline outputs from training tools."""
    run = (Path(root) / run).resolve()
    state = json.loads((run / 'stage_state.json').read_text())
    return RunPaths(root, run, state).restore_state()
