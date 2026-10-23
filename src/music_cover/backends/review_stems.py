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

def separate_review(request, output):
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from audio_separator.separator import Separator

    output = Path(output).resolve()
    stems = output / "audio"
    stems.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(int(request.get("seed", 42)))
    torch.manual_seed(int(request.get("seed", 42)))
    metrics = json.loads(Path(request["mix_metrics"]).read_text())
    master, voices, accompaniment, residual = mix_components(
        request["final"], request["vocal"], request["instrumental"], metrics)
    frames = len(master)
    for role, data in [("vocals", voices), ("instrumental", accompaniment), ("mix_residual", residual)]:
        write_stem(stems, role, data)
    del residual

    print("[review] Separating converted vocal into estimated lead/backing", flush=True)
    raw = output / "_karaoke"
    raw.mkdir()
    old_cwd = Path.cwd()
    try:
        os.chdir(raw)
        separator = Separator(model_file_dir=str(ROOT / "models/separation"), output_dir=str(raw),
                              output_format="WAV", normalization_threshold=1.0,
                              amplification_threshold=0.0, sample_rate=44100, use_soundfile=True,
                              mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                                           "batch_size": 1, "overlap": 4, "pitch_shift": 0})
        separator.load_model(model_filename=KARAOKE)
        separator.separate(str(stems / "vocals.wav"), {"Vocals": "lead", "Instrumental": "backing"})
        lead = stereo_aligned(raw / "lead.wav", frames)
        write_stem(stems, "lead_vocals", lead)
        # The karaoke model predicts lead; remainder can include reverb and bleed.
        write_stem(stems, "backing_vocals", voices - lead)
        del separator, lead
    finally:
        os.chdir(old_cwd)
    shutil.rmtree(raw)
    gc.collect()
    torch.cuda.empty_cache()

    print("[review] Separating final accompaniment using Demucs 6s", flush=True)
    model = get_model("htdemucs_6s", repo=ROOT / "models/review/demucs").eval().to("cuda")
    wave = torch.from_numpy(read_at(stems / "instrumental.wav", model.samplerate).T.copy())
    ref = wave.mean(0)
    mean, std = ref.mean(), ref.std().clamp_min(1e-8)
    with torch.inference_mode():
        estimates = apply_model(model, ((wave - mean) / std)[None], device="cuda", shifts=1,
                                split=True, overlap=0.25, progress=True, num_workers=0)[0].cpu()
    estimates = estimates * std + mean
    raw = output / "_demucs"
    raw.mkdir()
    reconstructed = np.zeros_like(accompaniment)
    for index, source in enumerate(model.sources):
        role = "instrumental_vocal_leak" if source == "vocals" else source
        temporary = raw / f"{role}.wav"
        sf.write(temporary, estimates[index].T.numpy(), model.samplerate, subtype="FLOAT")
        data = stereo_aligned(temporary, frames)
        write_stem(stems, role, data)
        reconstructed += data
    write_stem(stems, "instrumental_residual", accompaniment - reconstructed)
    shutil.rmtree(raw)
    del model, estimates, wave, reconstructed
    gc.collect()
    torch.cuda.empty_cache()

    total = np.zeros_like(master, dtype=np.float64)
    entries = {}
    for role in ("vocals", "instrumental", *LEAF_ROLES):
        path = stems / f"{role}.wav"
        info = inspect_audio(path)
        if info["frames"] != frames or info["sample_rate"] != RATE or info["channels"] != 2:
            raise RuntimeError(f"Unaligned review stem: {role}")
        info.update(path=f"audio/{role}.wav", label=LABELS[role],
                    sum_with_leaf_tracks=role in LEAF_ROLES,
                    status="near_silent" if info["rms"] < 1e-5 else "audio_present_not_quality_verified")
        entries[role] = info
        if role in LEAF_ROLES:
            total += sf.read(path, dtype="float64", always_2d=True)[0]
    error = total - master
    maximum = float(np.max(np.abs(error)))
    if maximum > 2e-6:
        raise RuntimeError(f"Leaf stems do not reconstruct final.wav: max error {maximum}")
    sf.write(output / "stems_recombined.wav", total, RATE, subtype="FLOAT")
    lock = json.loads((ROOT / "configs/review_models.lock.json").read_text())
    save_json(output / "stems.json", {
        "schema_version": 1, "input_final_sha256": sha256(request["final"]),
        "sample_rate": RATE, "frames": frames, "duration_s": frames / RATE,
        "time_origin": "sample 0 of final.wav; leading silence preserved", "stems": entries,
        "leaf_sum_roles": list(LEAF_ROLES),
        "reconstruction": {"max_abs_error": maximum, "rms_error": float(np.sqrt(np.mean(error ** 2))),
                           "includes_residuals": True, "proves_separation_quality": False},
        "mix_gains": {"master_gain": metrics["final"]["master_gain"], "vocal_gain": metrics["vocal_gain"]},
        "models": lock,
        "limitations": ["Model estimates, not original recording stems; leakage is possible.",
                        "Backing vocal estimate can contain lead reverb or artifacts, not proof of harmony singers.",
                        "Other can contain synths/strings; a named stem does not prove that instrument is present.",
                        "Parent vocals/instrumental must not be summed together with their child tracks.",
                        "Residuals preserve reconstruction; they are not musical instruments."]})
