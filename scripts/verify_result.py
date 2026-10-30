#!/usr/bin/env python3
"""Compatibility entry for the archived v0.1 diagnostic; see scripts/legacy/."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent / 'legacy/verify_result_v01.py'), run_name='__main__')
