from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from ..common.paths import ROOT

def transcribe(args):
    import torch
    from transformers import AutoModel
    torch.set_num_threads(4)
    model = AutoModel.from_pretrained(
        str(ROOT / "models/SheetSage2"), trust_remote_code=True, local_files_only=True,
        base_model_path=str(ROOT / "models/MERT-v2-FullSong"),
    ).eval().to("cuda")
    result = model.transcribe(args.input, output_dir=args.output, melody_only=args.stage != "transcribe_final", dtype="bf16",
                              preset="default", progress=lambda event: print(event, flush=True))
    if args.stage == "transcribe_final":
        import mido
        files = {}
        for path in Path(args.output).glob("*.mid"):
            midi = mido.MidiFile(path)
            files[path.name] = {"note_count": sum(m.type == "note_on" and m.velocity > 0 for t in midi.tracks for m in t),
                                "duration_s": midi.length}
        if not files.get("transcription.mid", {}).get("note_count"):
            raise RuntimeError("Final transcription contains no notes")
        (Path(args.output) / "transcription_summary.json").write_text(json.dumps({
            "source": str(args.input), "melody_only": False, "midis": files,
            "abc_error": result.get("abc_error"), "status": "model_estimates_require_human_review"}, indent=2) + "\n")
    elif result.get("abc_error") or not result.get("abc"):
        raise RuntimeError(result.get("abc_error") or "No ABC was produced")
