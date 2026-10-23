#!/usr/bin/env python3
"""Compatibility CLI for isolated backend calls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_cover.backends.yue import cover_token_budget
from music_cover.backends.cli import main

if __name__ == "__main__":
    main()
