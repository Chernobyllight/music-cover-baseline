"""Portable RVC training metadata and temporary upstream file lists."""
from copy import deepcopy
from pathlib import Path
from .voice_paths import portable_path


def portable_config(config, root):
    result = deepcopy(config)
    for key in ('source_dataset', 'dataset_dir', 'export_dir'):
        result[key] = portable_path((Path(root) / Path(result[key]).expanduser()).resolve(), root)
    for case in result.get('validation', {}).get('cases', []):
        case['source_run'] = portable_path((Path(root) / Path(case['source_run']).expanduser()).resolve(), root)
    return result


def runtime_filelist(folder, destination, path_base=None):
    """Resolve dataset-relative lists without changing their hashed source.

    Legacy lists used one absolute dataset root. Recover it from the exact
    train/<clip>.wav layout, then relocate all four paths against that root.
    Unknown external paths and paths escaping the current dataset are rejected.
    """
    folder = Path(folder).resolve()
    rows = [line.split('|') for line in (folder / 'train.txt').read_text().splitlines() if line.strip()]
    if not rows or any(len(row) != 5 or row[-1] != '0' for row in rows):
        raise ValueError('Invalid RVC training file list')
    old_root = None
    if path_base is None:
        first = Path(rows[0][0])
        if not first.is_absolute() or first.parent.name != 'train':
            raise ValueError('Unrecognized legacy RVC file list')
        old_root = first.parent.parent
    elif path_base != 'dataset':
        raise ValueError('Unknown RVC file-list path base')
    result = []
    for row in rows:
        paths = []
        for value in row[:4]:
            file = Path(value)
            if old_root is not None:
                try:
                    file = file.relative_to(old_root)
                except ValueError as error:
                    raise ValueError('Legacy RVC file list mixes dataset roots') from error
            elif file.is_absolute():
                raise ValueError('Dataset-relative RVC file list contains an absolute path')
            target = (folder / file).resolve()
            if not target.is_relative_to(folder) or not target.is_file():
                raise ValueError(f'RVC file-list input missing/outside dataset: {value}')
            paths.append(str(target))
        result.append('|'.join([*paths, row[-1]]))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(result) + '\n')
    return destination
