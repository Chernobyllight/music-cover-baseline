#!/usr/bin/env python3
"""Read-only source checks; does not load models or rewrite runtime locks."""
import argparse
import ast
import datetime
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
TREES = ('src', 'scripts', 'tests', 'configs', 'docs')


def source_files(root):
    roots = [p for p in root.iterdir() if p.is_file()]
    return sorted(roots + [p for name in TREES for p in (root / name).rglob('*')
                           if p.is_file() and '__pycache__' not in p.parts])


def check(root=ROOT):
    root = Path(root).resolve()
    errors = []
    counts = dict(python=0, shell=0, config=0, markdown=0, local_links=0)
    for file in source_files(root):
        relative = str(file.relative_to(root))
        if file.name.startswith('._') or file.name == '.DS_Store':
            errors.append(f'{relative}: operating-system metadata must not enter the release')
            continue
        try:
            if file.suffix == '.py':
                ast.parse(file.read_text(), filename=relative)
                counts['python'] += 1
            elif file.suffix == '.sh':
                result = subprocess.run(['bash', '-n', str(file)], capture_output=True, text=True, timeout=15)
                if result.returncode:
                    errors.append(f'{relative}: {result.stderr.strip()}')
                counts['shell'] += 1
            elif 'configs' in file.relative_to(root).parts and file.suffix in {'.json', '.yaml', '.yml'}:
                value = json.loads(file.read_text()) if file.suffix == '.json' else yaml.safe_load(file.read_text())
                if not isinstance(value, (dict, list)):
                    errors.append(f'{relative}: configuration must be a mapping or list')
                counts['config'] += 1
            elif file.suffix == '.md':
                counts['markdown'] += 1
                for match in re.finditer(r'\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)', file.read_text()):
                    url = urlsplit(match.group(1).strip('<>'))
                    if url.scheme or url.netloc or not url.path:
                        continue
                    target = (file.parent / unquote(url.path)).resolve()
                    counts['local_links'] += 1
                    if not target.exists():
                        errors.append(f'{relative}: missing linked file {match.group(1)}')
        except (OSError, ValueError, SyntaxError, yaml.YAMLError, subprocess.SubprocessError) as error:
            errors.append(f'{relative}: {error}')
    pipeline = yaml.safe_load((root / 'configs/pipeline.yaml').read_text())
    for stage, filename in pipeline['module_configs'].items():
        for file in [root / filename, root / 'scripts' / stage / 'run.sh', root / 'scripts' / stage / 'README.md']:
            if not file.is_file():
                errors.append(f'{stage}: missing module entry/config/README {file}')
    return dict(status='passed' if not errors else 'failed',
                checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                counts=counts, errors=errors,
                scope='source syntax, local documentation links, config parsing and stage entry files; GPU inference is a separate acceptance check')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Save report outside published run artifacts, e.g. cache/source_check.json')
    args = parser.parse_args()
    result = check()
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    raise SystemExit(bool(result['errors']))


if __name__ == '__main__':
    main()
