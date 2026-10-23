#!/usr/bin/env python3
"""Thin process-isolated adapters for the pinned upstream inference projects."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cover_token_budget(score_seconds, prefix_tokens, context=24576, tokens_per_second=25):
    """Leave room for a complete score and ending, without shortening for context."""
    if not math.isfinite(score_seconds) or score_seconds <= 0:
        raise ValueError("Score duration must be positive and finite")
    margin = max(30.0, score_seconds * 0.15)
    maximum = max(9000, math.ceil((score_seconds + margin) * tokens_per_second / 1000) * 1000)
    if prefix_tokens + maximum > context:
        raise ValueError(
            f"Score needs {maximum} semantic tokens plus {prefix_tokens} prompt tokens, "
            f"exceeding YuE2 context {context}; refusing to silently truncate the song")
    return {"score_duration_s": score_seconds, "ending_margin_s": margin,
            "semantic_tokens_per_second": tokens_per_second, "semantic_max_tokens": maximum,
            "audio_ceiling_s": maximum / tokens_per_second, "prefix_tokens": prefix_tokens,
            "context_tokens": context}


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


def cover(args):
    from yue2 import YuE2Pipeline
    sys.path.insert(0, str(ROOT / "src"))
    from music_cover.pitch import abc_tools
    request = json.loads(Path(args.request).read_text())
    request["abc"] = Path(args.score).read_text()
    request["cot"] = "melody"
    score = abc_tools().parse(request["abc"])
    score_seconds = float(score.voices["Vocal"].time * 60 / score.bpm)
    with YuE2Pipeline.from_pretrained(
        str(ROOT / "models/YuE2-3B"), vae=str(ROOT / "models/YuE2-Vae"),
        device="cuda", local_files_only=True, memory_budget_gib=80,
    ) as pipeline:
        # Planning from an external score only tokenizes it; it performs no generation.
        plan = pipeline.plan(**request)
        budget = cover_token_budget(score_seconds, len(plan.prefix), pipeline.generation_config.context)
        (Path(args.output) / "generation_budget.json").write_text(json.dumps(budget, indent=2) + "\n")
        print(f"[length] score={score_seconds:.2f}s, semantic max_tokens={budget['semantic_max_tokens']} "
              f"(audio ceiling {budget['audio_ceiling_s']:.0f}s)", flush=True)
        result = pipeline(**request, semantic_sampling={"max_tokens": budget["semantic_max_tokens"]})
        result.save_artifacts(Path(args.output) / "generated")
        if any(result.truncated.values()):
            raise RuntimeError(f"YuE2 result truncated: {result.truncated}")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["transcribe", "transcribe_final", "cover", "separate", "convert", "pitch",
                                         "review_stems", "review_midi", "review_package"])
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--request")
    parser.add_argument("--score")
    parser.add_argument("--reference")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    start = time.monotonic()
    if args.stage.startswith("review_"):
        sys.path.insert(0, str(ROOT / "src"))
        from music_cover.review import run_stage
        run_stage(args)
    elif args.stage == "transcribe_final":
        transcribe(args)
    else:
        globals()[args.stage](args)
    print(json.dumps({"stage": args.stage, "wall_seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
