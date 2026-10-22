from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_audio(path):
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    peak = float(np.max(np.abs(data)))
    return {"sample_rate": rate, "channels": data.shape[1], "frames": len(data),
            "duration_s": len(data) / rate, "sample_peak": peak,
            "rms": float(np.sqrt(np.mean(data.astype(np.float64) ** 2))),
            "subtype": sf.info(path).subtype, "sha256": sha256(path)}


def decode(source, destination, *, rate=48000, channels=2, start=0.0, duration=None):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if start:
        command += ["-ss", str(start)]
    command += ["-i", str(source), "-vn"]
    if duration is not None:
        command += ["-t", str(duration)]
    command += ["-ar", str(rate), "-ac", str(channels), "-c:a", "pcm_f32le", str(destination)]
    subprocess.run(command, check=True, timeout=600)
    return inspect_audio(destination)


def read_at(path, rate=48000):
    data, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError(f"Invalid audio: {path}")
    if source_rate != rate:
        divisor = math.gcd(rate, source_rate)
        data = signal.resample_poly(data, rate // divisor, source_rate // divisor, axis=0)
    return data.astype(np.float32, copy=False)


def fit_length(data, frames):
    if len(data) >= frames:
        return data[:frames]
    return np.pad(data, ((0, frames - len(data)), (0, 0)))


def shift_audio(data, shift_samples, frames):
    """Positive shift delays the audio; negative shift advances it."""
    if shift_samples > 0:
        data = np.pad(data, ((shift_samples, 0), (0, 0)))
    elif shift_samples < 0:
        data = data[-shift_samples:]
    return fit_length(data, frames)


def envelope(data, hop):
    mono = data.mean(axis=1)
    count = len(mono) // hop
    if not count:
        return np.zeros(0, dtype=np.float64)
    return np.sqrt(np.mean(mono[:count * hop].reshape(count, hop).astype(np.float64) ** 2, axis=1))


def estimate_alignment(source, converted, rate):
    hop = round(rate * 0.01)
    first, second = envelope(source, hop), envelope(converted, hop)
    length = min(len(first), len(second))
    width = min(length, 2000)
    rows = []
    for start in sorted(set([0, max(0, (length - width) // 2), max(0, length - width)])):
        a, b = first[start:start + width], second[start:start + width]
        if len(a) < 20 or max(float(a.max()), float(b.max())) < 1e-5:
            continue
        a, b = a - a.mean(), b - b.mean()
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm < 1e-12:
            continue
        corr = signal.correlate(b, a, mode="full", method="fft") / norm
        lags = signal.correlation_lags(len(b), len(a), mode="full")
        keep = np.abs(lags) <= 25
        offset = int(np.argmax(corr[keep]))
        lag, confidence = int(lags[keep][offset]), float(corr[keep][offset])
        rows.append({"start_s": start * hop / rate, "lag_ms": lag * hop / rate * 1000,
                     "correlation": confidence})
    accepted = [row["lag_ms"] for row in rows if row["correlation"] >= 0.3]
    consistent = bool(accepted) and max(accepted) - min(accepted) <= 30
    delay_ms = float(np.median(accepted)) if consistent else 0.0
    return {"windows": rows, "consistent": consistent, "estimated_delay_ms": delay_ms,
            "applied_shift_samples": -round(delay_ms * rate / 1000),
            "needs_listening_review": not consistent or abs(delay_ms) >= 200}


def oversampled_peak(data):
    # Four-times oversampling is an engineering estimate, not a certified meter.
    return float(np.max(np.abs(signal.resample_poly(data, 4, 1, axis=0))))


def export_limited(path, data, rate=48000, ceiling_db=-1.1):
    peak = oversampled_peak(data)
    limit = 10 ** (ceiling_db / 20)
    gain = min(1.0, limit / max(peak, 1e-12))
    sf.write(path, data * gain, rate, subtype="PCM_24")
    result = inspect_audio(path)
    written, _ = sf.read(path, dtype="float32", always_2d=True)
    measured = oversampled_peak(written)
    result.update({"master_gain": gain, "estimated_true_peak_dbtp": 20 * math.log10(max(measured, 1e-12)),
                   "peak_method": "scipy.resample_poly 4x; engineering estimate"})
    return result


def mix_tracks(cover, vocal, instrumental, converted, output):
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
    duration_delta = (len(replacement) - len(source)) / rate
    if abs(duration_delta) > 0.25:
        raise ValueError(f"Converted vocal differs by {duration_delta:.3f}s; refusing to hide drift")
    alignment = estimate_alignment(source, replacement, rate)
    source = fit_length(source, frames)
    accompaniment = fit_length(accompaniment, frames)
    replacement = shift_audio(replacement, alignment["applied_shift_samples"], frames)
    replacement = replacement.mean(axis=1, keepdims=True)
    source_mono = source.mean(axis=1)
    active = np.abs(source_mono) > max(float(np.max(np.abs(source_mono))) * 0.01, 1e-5)
    source_rms = float(np.sqrt(np.mean(source_mono[active].astype(np.float64) ** 2))) if active.any() else 0.0
    converted_rms = float(np.sqrt(np.mean(replacement[active, 0].astype(np.float64) ** 2))) if active.any() else 0.0
    if source_rms < 1e-5 or converted_rms < 1e-5:
        raise ValueError("Source or converted vocal is effectively silent")
    vocal_gain = float(np.clip(source_rms / converted_rms, 0.25, 4.0))
    reconstruction = export_limited(output / "reconstruction.wav", source + accompaniment, rate)
    final = export_limited(output / "final.wav", accompaniment + replacement * vocal_gain, rate)
    sf.write(output / "converted_aligned.wav", replacement, rate, subtype="FLOAT")
    alignment.update({"converted_duration_delta_s": duration_delta, "output_frames": frames})
    return {"alignment": alignment, "vocal_gain": vocal_gain,
            "reconstruction": reconstruction, "final": final,
            "subjective_quality": "not_listened; human listening required"}
