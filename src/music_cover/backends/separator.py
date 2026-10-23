from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from ..common.paths import ROOT

def separate(args):
    from audio_separator.separator import Separator
    # audio-separator 0.30.2's soundfile writer uses the process working
    # directory, even when output_dir is provided. Keep every output in
    # this attempt's private directory so concurrent runs cannot collide.
    os.chdir(args.output)
    separator = Separator(
        output_dir=args.output, output_format="WAV", model_file_dir=str(ROOT / "models/separation"),
        normalization_threshold=1.0, amplification_threshold=0.0, sample_rate=44100,
        use_soundfile=True,
    )
    separator.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    results = separator.separate(args.input)
    found = {}
    for name in results:
        path = Path(name)
        if not path.is_absolute():
            path = Path(args.output) / path
        role = "instrumental" if "instrumental" in path.name.lower() else "vocals" if "vocals" in path.name.lower() else None
        if role:
            target = Path(args.output) / f"{role}.wav"
            if target != path:
                shutil.move(path, target)
            found[role] = str(target)
    if set(found) != {"instrumental", "vocals"}:
        raise RuntimeError(f"Unexpected separation outputs: {results}")
