"""Shared file integrity checks for exported singer profiles.

Backend modules remain responsible for schema, architecture and asset identity.
This helper does not load model weights or change checkpoint selection.
"""
from pathlib import Path

from ..audio import sha256


def verified_files(profile_path, profile, keys, *, label='Voice profile', contained=False):
    directory = Path(profile_path).resolve().parent
    files = {}
    for key in keys:
        item = profile[key]
        path = (directory / item['path']).resolve()
        if contained and not path.is_relative_to(directory):
            raise ValueError(f'{label} {key} checksum/path mismatch: {path}')
        if not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError(f'{label} {key} missing or checksum mismatch: {path}')
        files[key] = path
    return files
