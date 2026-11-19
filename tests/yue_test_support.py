"""Locate the YuE source that the default Mac launcher installs."""
import os
from pathlib import Path

from music_cover.common.macos import WORKSPACE

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("MUSIC_TEST_DEVICE", "cpu")


def require_assembled_yue(module):
    expected = (ROOT / WORKSPACE / "third_party/YuE/src/yue2").resolve()
    actual = Path(module.__file__).resolve()
    if not actual.is_relative_to(expected):
        raise AssertionError(f"MPS regression requires YuE from {expected}; imported {actual}")
