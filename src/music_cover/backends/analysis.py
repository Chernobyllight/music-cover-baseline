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
from ..common.paths import ROOT
from ..common.io import write_json as save_json, artifact_paths

from ..common.review_audio import RATE, KARAOKE, LEAF_ROLES, PITCHED_ROLES, LABELS, stereo_aligned, mix_components, write_stem

def loudness(path):
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path),
               "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
    start, end = result.stderr.rfind("{"), result.stderr.rfind("}")
    raw = json.loads(result.stderr[start:end + 1])
    return {key: float(raw[value]) if math.isfinite(float(raw[value])) else None for key, value in
            [("integrated_lufs", "input_i"), ("true_peak_dbtp", "input_tp"), ("loudness_range_lu", "input_lra")]}

def describe_audio(path):
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, units="time")
    return {"loudness": loudness(path), "estimated_tempo_bpm": float(np.asarray(tempo).reshape(-1)[0]),
            "estimated_beat_count": len(beats), "spectral_centroid_mean_hz": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
            "onset_strength_mean": float(onset.mean()), "audio_sha256": sha256(path),
            "note": "Descriptive metrics only; do not prove style change, musical quality, or voice identity."}
