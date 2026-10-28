from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from ..common.paths import ROOT
from ..common.io import write_json
from ..audio import sha256
from ..voice import resolve_voice, voice_record
from ..voice_weights import check_parameter_coverage

def convert(args):
    voice = resolve_voice(getattr(args, 'voice_mode', 'zero_shot'), getattr(args, 'voice_profile', None))
    # Candidate checkpoints are accepted only by the isolated validation controller.
    if getattr(args, 'checkpoint', None):
        voice = {**voice, 'mode': 'candidate', 'checkpoint': Path(args.checkpoint).resolve(),
                 'config': Path(args.model_config).resolve()}
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
    original_loader = inference.load_checkpoint
    def checked_loader(model, optimizer, path, **kwargs):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('adaptation_scope') not in (None, 'speaker'):
            raise ValueError('Only speaker-level checkpoints are supported')
        state = payload['net']
        check_parameter_coverage(model, state)
        del state, payload
        return original_loader(model, optimizer, path, **kwargs)
    inference.load_checkpoint = checked_loader
    options = argparse.Namespace(
        source=args.input, target=args.reference, output=args.output,
        diffusion_steps=args.steps, length_adjust=1.0, inference_cfg_rate=0.7,
        f0_condition=True, auto_f0_adjust=False, semi_tone_shift=0,
        checkpoint=str(voice['checkpoint']), config=str(voice['config']), fp16=True,
    )
    inference.main(options)
    outputs = list(Path(args.output).glob("vc_*.wav"))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one conversion WAV, got {outputs}")
    if outputs[0].name != "converted.wav":
        outputs[0].rename(Path(args.output) / "converted.wav")
    write_json(Path(args.output) / 'conversion.json', {**voice_record(voice),
        'source_sha256': sha256(args.input), 'reference_sha256': sha256(args.reference),
        'steps': args.steps, 'seed': args.seed, 'semitones': 0, 'auto_f0_adjust': False,
        'training_performed': False, 'reference_required': True})
