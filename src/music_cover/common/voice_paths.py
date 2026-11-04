"""Portable Seed-VC metadata; absolute paths are confined to execution records."""
import copy
import json
from pathlib import Path
import yaml
from .paths import ROOT


def project_path(value, root=None):
    root = Path(root or ROOT)
    value = Path(value).expanduser()
    return value if value.is_absolute() else root / value


def portable_path(value, root=None):
    root = Path(root or ROOT)
    value = project_path(value, root)
    try:
        return value.relative_to(root).as_posix()
    except ValueError:
        return str(value)


def training_output(run, stage):
    """Read new run-relative state and legacy absolute stage artifact paths."""
    run = Path(run)
    state = json.loads((run / 'state.json').read_text())
    value = Path(state['stages'][stage]['output'])
    if not value.is_absolute():
        return run / value
    # Exact stage layout, never a basename search or a mount-name replacement.
    tail = value.parts[-4:]
    if (len(tail) == 4 and tail[:2] == ('stages', stage)
            and tail[2].startswith('attempt_') and tail[2][8:].isdigit()
            and tail[3] == 'artifacts'):
        return run.joinpath(*tail)
    return value


def runtime_model_config(source, destination, root=None):
    """Resolve local assets in a separate copy, preserving exported file hashes.

    Old exports identify their former project root through log_dir's runs path.
    Existing external absolute assets remain external. Unknown missing paths fail.
    """
    root = Path(root or ROOT)
    cfg = copy.deepcopy(yaml.safe_load(Path(source).read_text()))
    log = Path(cfg.get('log_dir', 'runs'))
    old_root = None
    if log.is_absolute() and 'runs' in log.parts:
        old_root = Path(*log.parts[:log.parts.index('runs')])
    for name in ('vocoder', 'speech_tokenizer'):
        item = cfg['model_params'][name]
        value = Path(item['name']).expanduser()
        if not value.is_absolute():
            value = root / value
        elif not value.exists() and old_root is not None:
            try:
                value = root / value.relative_to(old_root)
            except ValueError:
                pass
        if not value.is_dir():
            raise FileNotFoundError(f'Seed-VC {name} local model directory missing: {value}')
        item['name'] = str(value)
    if not log.is_absolute():
        cfg['log_dir'] = str(root / log)
    elif old_root is not None:
        cfg['log_dir'] = str(root / log.relative_to(old_root))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return destination
