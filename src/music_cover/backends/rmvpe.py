from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from ..common.paths import ROOT

def pitch(args):
    import numpy as np
    import torch
    import librosa
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "third_party/seed-vc"))
    from music_cover.pitch import summarize_pitch
    from modules.rmvpe import RMVPE
    torch.set_num_threads(4)
    model = RMVPE(str(ROOT / "models/RMVPE/rmvpe.pt"), is_half=False, device="cuda")
    tracks = json.loads(Path(args.request).read_text())["tracks"]
    profiles = {"method": "RMVPE, 16kHz, 10ms hop; F0 65..1000Hz; RMS >= -40dBFS; no gain normalization",
                "tracks": {}}
    for name, path in tracks.items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Invalid pitch track name")
        samples, _ = librosa.load(path, sr=16000, mono=True)
        # Thirty-second cores with one second context bound memory on long songs.
        total = len(samples) // 160 + 1
        f0 = np.zeros(total, np.float32)
        for first in range(0, total, 3000):
            last = min(total, first + 3000)
            left, right = max(0, first - 100), min(total, last + 100)
            block = samples[left * 160:min(len(samples), right * 160)]
            estimated = model.infer_from_audio(block, thred=0.03)
            core = estimated[first - left:last - left]
            f0[first:first + len(core)] = core
        rms = librosa.feature.rms(y=samples, frame_length=1024, hop_length=160)[0]
        count = min(len(f0), len(rms))
        f0, rms = f0[:count], rms[:count]
        np.savez_compressed(Path(args.output) / f"{name}_f0.npz", f0=f0, rms=rms, hop_seconds=0.01)
        profiles["tracks"][name] = {
            "path": str(path), "duration_s": len(samples) / 16000,
            "whole": summarize_pitch(f0, rms),
            "windows": [{"start_s": start, "end_s": min(start + 30, len(samples) / 16000),
                         **summarize_pitch(f0, rms, start, start + 30)}
                        for start in range(0, int(len(samples) / 16000), 30)],
        }
        print(name, profiles["tracks"][name]["whole"], flush=True)
    (Path(args.output) / "pitch_profiles.json").write_text(json.dumps(profiles, ensure_ascii=False, indent=2) + "\n")
