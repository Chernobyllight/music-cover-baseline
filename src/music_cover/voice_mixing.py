"""Voice result mixing with advisory quality checks and explicit timing policy."""
from pathlib import Path
import numpy as np
import soundfile as sf
from .audio import read_at, fit_length, shift_audio, export_limited, export_mp3
from .voice_alignment import assess_alignment


def mix_tracks(cover, vocal, instrumental, converted, output, *, mp3_bitrate_kbps=192):
    output = Path(output)
    rate = 48000
    master = read_at(cover, rate)
    source, accompaniment, replacement = [read_at(p, rate) for p in (vocal, instrumental, converted)]
    frames = len(master)
    if master.shape[1] != 2 or accompaniment.shape[1] != 2:
        raise ValueError("Cover and accompaniment must be stereo")
    for name, wave in [("source vocal", source), ("accompaniment", accompaniment)]:
        if abs(len(wave) - frames) / rate > 0.1:
            raise ValueError(f"{name} differs from cover by more than 100 ms")
    if not len(replacement) or not np.isfinite(replacement).all():
        raise ValueError('Converted audio is empty or nonfinite; no readable result to mix')
    duration_delta = (len(replacement) - len(source)) / rate
    warnings = []
    if abs(duration_delta) > .25:
        warnings.append('duration_mismatch; preserve longer timeline and pad shorter tracks with silence')
    frames = max(frames, len(replacement))
    alignment = assess_alignment(source, replacement, rate)
    source = fit_length(source, frames)
    accompaniment = fit_length(accompaniment, frames)
    replacement = shift_audio(replacement, alignment["applied_shift_samples"], frames)
    replacement = replacement.mean(axis=1, keepdims=True)
    source_mono = source.mean(axis=1)
    active = np.abs(source_mono) > max(float(np.max(np.abs(source_mono))) * 0.01, 1e-5)
    source_rms = float(np.sqrt(np.mean(source_mono[active].astype(np.float64) ** 2))) if active.any() else 0.0
    converted_rms = float(np.sqrt(np.mean(replacement[active, 0].astype(np.float64) ** 2))) if active.any() else 0.0
    if source_rms < 1e-5 or converted_rms < 1e-5:
        warnings.append('source_or_converted_vocal_effectively_silent')
        vocal_gain = 1.0
    else:
        vocal_gain = float(np.clip(source_rms / converted_rms, 0.25, 4.0))
    if not alignment['consistent']:
        warnings.append('alignment_uncertain; no global shift applied')
    reconstruction = export_limited(output / "reconstruction.wav", source + accompaniment, rate)
    final = export_limited(output / "final.wav", accompaniment + replacement * vocal_gain, rate)
    final_mp3 = export_mp3(output / "final.wav", output / "final.mp3", bitrate_kbps=mp3_bitrate_kbps)
    sf.write(output / "converted_aligned.wav", replacement, rate, subtype="FLOAT")
    alignment.update({"converted_duration_delta_s": duration_delta, "output_frames": frames})
    return {"alignment": alignment, "vocal_gain": vocal_gain, "warnings": warnings,
            "generation_status":"succeeded", "delivery_requires_review":bool(warnings),
            "duration_policy":"preserve longest track; silence padding; no time stretching",
            "reconstruction": reconstruction, "final": final, "final_mp3": final_mp3,
            "subjective_quality": "not_listened; human listening required"}
