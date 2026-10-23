from __future__ import annotations
import csv
import gc
import html
import json
import math
import os
import random
import shutil
import subprocess
import zipfile
from pathlib import Path
import numpy as np
import soundfile as sf
from ..audio import fit_length, inspect_audio, read_at, sha256
from .paths import ROOT
from .io import write_json as save_json, artifact_paths

RATE = 48000
KARAOKE = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
LEAF_ROLES = ("lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano",
              "other", "instrumental_vocal_leak", "instrumental_residual", "mix_residual")
PITCHED_ROLES = {"lead_vocals": 53, "backing_vocals": 52, "vocals": 53,
                 "bass": 33, "guitar": 27, "piano": 0, "other": 89}
LABELS = {"vocals": "转换后总人声（父轨）", "instrumental": "最终混音伴奏（父轨）",
          "lead_vocals": "主唱估计", "backing_vocals": "和声／背景人声估计",
          "drums": "鼓与打击乐", "bass": "贝斯", "guitar": "吉他", "piano": "钢琴",
          "other": "其他乐器（可含合成器、弦乐等）",
          "instrumental_vocal_leak": "伴奏中被模型判为人声的残留",
          "instrumental_residual": "乐器分离重建残差", "mix_residual": "混音量化残差"}


def stereo_aligned(path, frames, rate=RATE):
    data = read_at(path, rate)
    if abs(len(data) - frames) > round(rate * 0.1):
        raise ValueError(f"Review stem duration differs by >100 ms: {path}")
    data = fit_length(data, frames)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    if data.shape[1] != 2:
        raise ValueError(f"Expected mono/stereo audio: {path}")
    return data

def mix_components(final, vocal, instrumental, metrics):
    """Recover the actual post-gain mix components, including PCM quantization."""
    master = read_at(final, RATE)
    n = len(master)
    gain = float(metrics["final"]["master_gain"])
    voices = stereo_aligned(vocal, n) * (gain * float(metrics["vocal_gain"]))
    backing = stereo_aligned(instrumental, n) * gain
    return master, voices, backing, master - voices - backing

def write_stem(directory, role, data):
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError(f"Invalid stem: {role}")
    path = Path(directory) / f"{role}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Float preserves headroom/cancellation. Do not normalize each stem separately.
    sf.write(path, data, RATE, subtype="FLOAT")
    return path
