from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from ..common.paths import ROOT

def convert(args):
    # Keep upstream relative asset lookup intact. All changes stay inside this project.
    project = ROOT / "third_party/seed-vc"
    os.chdir(project)
    sys.path.insert(0, str(project))
    import inference
    import torch
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    local_auxiliary = {
        ("lj1995/VoiceConversionWebUI", "rmvpe.pt"): ROOT / "models/RMVPE/rmvpe.pt",
        ("funasr/campplus", "campplus_cn_common.bin"): ROOT / "models/CAMPPlus/campplus_cn_common.bin",
    }
    def local_asset(repo_id, model_filename="pytorch_model.bin", config_filename=None):
        if config_filename is not None or (repo_id, model_filename) not in local_auxiliary:
            raise ValueError(f"Unregistered voice model asset: {repo_id}/{model_filename}")
        return str(local_auxiliary[repo_id, model_filename])
    inference.load_custom_model_from_hf = local_asset
    options = argparse.Namespace(
        source=args.input, target=args.reference, output=args.output,
        diffusion_steps=args.steps, length_adjust=1.0, inference_cfg_rate=0.7,
        f0_condition=True, auto_f0_adjust=False, semi_tone_shift=0,
        checkpoint=str(ROOT / "models/Seed-VC/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"),
        config=str(ROOT / "configs/seed_svc.local.yml"), fp16=True,
    )
    inference.main(options)
    outputs = list(Path(args.output).glob("vc_*.wav"))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one conversion WAV, got {outputs}")
    if outputs[0].name != "converted.wav":
        outputs[0].rename(Path(args.output) / "converted.wav")
