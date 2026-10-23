#!/usr/bin/env python3
"""Compatibility entrypoint; implementation moved to scripts/setup/."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent / 'setup/download_models.py'), run_name='__main__')
