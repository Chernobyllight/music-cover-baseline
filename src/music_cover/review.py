"""Post-conversion audio stems and explicitly labelled estimated MIDI for review.

Inference runs in envs/separator; importing this module needs only control's deps.
Parent stems are alternatives to their children, never additional mix tracks.
"""
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

from .audio import fit_length, inspect_audio, read_at, sha256

ROOT = Path(__file__).resolve().parents[2]
RATE = 48000
KARAOKE = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
LEAF_ROLES = ("lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano",
              "other", "instrumental_vocal_leak", "instrumental_residual", "mix_residual")
PITCHED_ROLES = {"lead_vocals": 53, "backing_vocals": 52, "vocals": 53,
                 "bass": 33, "guitar": 27, "piano": 0, "other": 89}
LABELS = {"vocals": "转换后总人声（父轨）", "instrumental": "最终混音伴奏（父轨）",
          "lead_vocals": "主唱估计", "backing_vocals": "和声／背景人声估计",
          "drums": "鼓与打击乐", "bass": "贝斯", "guitar": "吉他", "piano": "钢琴",
          "other": "其他乐器（可含合成器、弦乐等）",
          "instrumental_vocal_leak": "伴奏中被模型判为人声的残留",
          "instrumental_residual": "乐器分离重建残差", "mix_residual": "混音量化残差"}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def artifact_paths(directory):
    """Include every payload in stage cache validation, including nested files."""
    directory = Path(directory)
    return {str(p.relative_to(directory)): p for p in sorted(directory.rglob("*"))
            if p.is_file() and p.stat().st_size and p.name not in
            {"process.log", "command.json", "result.json", "failure.json"}}


def stereo_aligned(path, frames, rate=RATE):
    data = read_at(path, rate)
    if abs(len(data) - frames) > round(rate * 0.1):
        raise ValueError(f"Review stem duration differs by >100 ms: {path}")
    data = fit_length(data, frames)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    if data.shape[1] != 2:
        raise ValueError(f"Expected mono/stereo audio: {path}")
    return data


def mix_components(final, vocal, instrumental, metrics):
    """Recover the actual post-gain mix components, including PCM quantization."""
    master = read_at(final, RATE)
    n = len(master)
    gain = float(metrics["final"]["master_gain"])
    voices = stereo_aligned(vocal, n) * (gain * float(metrics["vocal_gain"]))
    backing = stereo_aligned(instrumental, n) * gain
    return master, voices, backing, master - voices - backing


def write_stem(directory, role, data):
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError(f"Invalid stem: {role}")
    path = Path(directory) / f"{role}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Float preserves headroom/cancellation. Do not normalize each stem separately.
    sf.write(path, data, RATE, subtype="FLOAT")
    return path


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


def write_midi(path, tracks, duration):
    """Seconds-based MIDI, fixed 120 BPM timebase, no musical quantization."""
    import mido
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("MIDI duration must be positive")
    midi = mido.MidiFile(type=1, ticks_per_beat=960)
    end_tick = round(duration * 1920)
    conductor = mido.MidiTrack()
    conductor.extend([mido.MetaMessage("track_name", name="Absolute time; 120 BPM is a timebase"),
                      mido.MetaMessage("set_tempo", tempo=500000),
                      mido.MetaMessage("marker", text="Audio end", time=end_tick),
                      mido.MetaMessage("end_of_track")])
    midi.tracks.append(conductor)
    channels = iter([i for i in range(16) if i != 9])
    for spec in tracks:
        channel = 9 if spec.get("drums") else next(channels)
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=spec["name"]))
        track.append(mido.Message("program_change", program=spec.get("program", 0), channel=channel))
        events = []
        for note in spec["notes"]:
            a, b, pitch, velocity = note[:4]
            if not all(math.isfinite(float(v)) for v in (a, b, pitch, velocity)):
                raise ValueError("Non-finite MIDI note")
            if not (0 <= a < b <= duration + 1e-6 and 0 <= pitch <= 127 and 1 <= velocity <= 127):
                raise ValueError(f"Invalid MIDI note: {note}")
            first, last = round(a * 1920), min(end_tick, max(round(b * 1920), round(a * 1920) + 1))
            if last <= first:
                continue
            events += [(first, 1, int(pitch), int(velocity)), (last, 0, int(pitch), 0)]
        previous = 0
        for tick, on, pitch, velocity in sorted(events):
            track.append(mido.Message("note_on" if on else "note_off", channel=channel,
                                      note=pitch, velocity=velocity, time=tick - previous))
            previous = tick
        track.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - previous)))
        midi.tracks.append(track)
    midi.save(path)


def notes_csv(path, notes):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start_seconds", "end_seconds", "midi_pitch", "velocity", "activation_not_calibrated_confidence"])
        writer.writerows(notes)


def midi_stats(path, duration):
    import pretty_midi
    import mido
    parsed = pretty_midi.PrettyMIDI(str(path))
    notes = [n for instrument in parsed.instruments for n in instrument.notes]
    for n in notes:
        if not 0 <= n.start < n.end <= duration + 0.002:
            raise RuntimeError(f"MIDI notes outside final audio: {path}")
    return {"notes": len(notes), "midi_duration_s": mido.MidiFile(path).length,
            "first_note_s": min((n.start for n in notes), default=None),
            "last_note_s": max((n.end for n in notes), default=None),
            "status": "estimated_needs_review" if notes else "no_notes_detected"}


def drum_onset_notes(path):
    """Coarse onset candidates, NOT a trained drum transcription/classifier."""
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    magnitude = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freq = librosa.fft_frequencies(sr=sr, n_fft=1024)
    rows = []
    for lo, hi, pitch in [(30, 180, 36), (180, 4500, 38), (4500, 10000, 42)]:
        band = magnitude[(freq >= lo) & (freq < hi)].mean(axis=0)
        flux = np.maximum(0, np.diff(np.log1p(band), prepend=0))
        if np.max(flux, initial=0) < 1e-5:
            continue
        times = librosa.onset.onset_detect(onset_envelope=flux, sr=sr, hop_length=256,
                                           units="time", backtrack=False, delta=0.12, wait=5)
        for start in times:
            end = min(float(start) + 0.06, len(y) / sr)
            if end > start:
                rows.append([float(start), end, pitch, 90, ""])
    return sorted(rows)


def transcribe_stems(request, output):
    import onnxruntime as ort
    import pretty_midi
    from basic_pitch.inference import Model, predict
    output = Path(output)
    for directory in ("midi", "notes", "activations", "raw_midi"):
        (output / directory).mkdir(exist_ok=True)
    root = Path(request["stems"])
    duration = json.loads((root / "stems.json").read_text())["duration_s"]
    # Pin CPU threading rather than spawning one thread per physical server core.
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    model = Model.__new__(Model)
    model.model_type = Model.MODEL_TYPES.ONNX
    model_path = ROOT / "models/review/basic_pitch.onnx"
    model.model = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    summaries, combined = {}, []
    for role, program in PITCHED_ROLES.items():
        print(f"[review MIDI] {role}", flush=True)
        audio_path = root / "audio" / f"{role}.wav"
        # Do not amplify silent or weak stems to manufacture notes.
        rms = inspect_audio(audio_path)["rms"]
        events, rows = [], []
        if rms >= 1e-5:
            activations, raw_midi, events = predict(
                audio_path, model, onset_threshold=0.5, frame_threshold=0.3,
                minimum_note_length=90, minimum_frequency=30 if role == "bass" else None,
                maximum_frequency=700 if role == "bass" else None, multiple_pitch_bends=False,
                midi_tempo=120)
            np.savez_compressed(output / "activations" / f"{role}.npz", **activations)
            raw_midi.write(str(output / "raw_midi" / f"{role}.mid"))
            for start, end, pitch, amplitude, _bends in events:
                start, end = max(0.0, float(start)), min(duration, float(end))
                if end > start:
                    rows.append([start, end, int(pitch), int(np.clip(round(float(amplitude) * 127), 1, 127)),
                                 float(amplitude)])
        spec = {"name": role, "program": program, "notes": sorted(rows)}
        path = output / "midi" / f"{role}.mid"
        write_midi(path, [spec], duration)
        notes_csv(output / "notes" / f"{role}.csv", rows)
        summaries[role] = {**midi_stats(path, duration), "method": "Basic Pitch 0.4.0 ONNX",
                           "audio_sha256": sha256(audio_path), "program_is_audition_hint_only": True,
                           "file": f"midi/{role}.mid", "notes_csv": f"notes/{role}.csv"}
        if role != "vocals":  # Parent vocal would double the lead/backing notes.
            combined.append(spec)

    drum_notes = drum_onset_notes(root / "audio/drums.wav")
    drum_spec = {"name": "drums_onset_candidates", "drums": True, "notes": drum_notes}
    write_midi(output / "midi/drums_onsets.mid", [drum_spec], duration)
    notes_csv(output / "notes/drums_onsets.csv", drum_notes)
    summaries["drums_onsets"] = {**midi_stats(output / "midi/drums_onsets.mid", duration),
        "method": "Heuristic band-onset candidates; NOT classified drum performance",
        "file": "midi/drums_onsets.mid", "notes_csv": "notes/drums_onsets.csv",
        "gm_mapping": {"36": "low-band candidate", "38": "mid-band candidate", "42": "high-band candidate"}}
    combined.append(drum_spec)
    write_midi(output / "midi/arrangement_estimate.mid", combined, duration)
    summaries["arrangement_estimate"] = {**midi_stats(output / "midi/arrangement_estimate.mid", duration),
        "file": "midi/arrangement_estimate.mid", "method": "Combined leaf-part estimates, excluding duplicate parent vocals and chord guide"}

    # Chord guide comes from transcribing FINAL audio with melody_only=False.
    chord_path = Path(request["transcription"]) / "chords.mid"
    chords = pretty_midi.PrettyMIDI(str(chord_path))
    rows = [[max(0.0, n.start), min(duration, n.end), n.pitch, n.velocity, ""]
            for instrument in chords.instruments for n in instrument.notes
            if min(duration, n.end) > max(0.0, n.start)]
    write_midi(output / "midi/harmony.mid", [{"name": "Estimated chord guide", "program": 0, "notes": rows}], duration)
    notes_csv(output / "notes/harmony.csv", rows)
    summaries["harmony"] = {**midi_stats(output / "midi/harmony.mid", duration),
        "file": "midi/harmony.mid", "notes_csv": "notes/harmony.csv",
        "method": "SheetSage2 chord guide from final mix; voicing is not the original instrument performance"}
    save_json(output / "midi.json", {"schema_version": 1, "source": "post-conversion final mix and aligned stems",
        "duration_s": duration, "timebase_bpm": 120, "tempo_is_musical_estimate": False,
        "quantized": False, "audio_start_seconds": 0, "tracks": summaries,
        "model_sha256": sha256(model_path),
        "parameters": {"onset_threshold": 0.5, "frame_threshold": 0.3, "minimum_note_ms": 90},
        "limitations": ["All MIDI is automatically estimated; not ground truth or the generator's native score.",
                        "MIDI has notes, not the singer's timbre. GM programs are audition hints only.",
                        "120 BPM is a fixed seconds-to-ticks timebase; see final_transcription/beat.lab for estimated beats.",
                        "Main MIDI omits pitch bends to avoid conflicting bends in polyphony; raw_midi retains model output.",
                        "Drum files are onset candidates with approximate frequency-band labels, requiring manual correction.",
                        "Empty or near-silent parts remain explicit rather than receiving invented notes."]})


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


def build_review_package(request, output):
    output = Path(output)
    for key, directory in [("stems", ""), ("transcription", "final_transcription"), ("midi", "")]:
        source = Path(request[key])
        for rel, path in artifact_paths(source).items():
            destination = output / directory / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    source_transcription = Path(request["source_transcription"])
    target = output / "source_transcription"
    target.mkdir()
    for path in source_transcription.iterdir():
        if path.suffix in {".mid", ".abc", ".lab"} and path.is_file() and path.stat().st_size:
            shutil.copyfile(path, target / path.name)
    save_json(output / "request.json", request["run_request"])
    lyric_source = request["run_request"].get("lyrics")
    if lyric_source:
        shutil.copyfile(lyric_source, output / "lyrics_input.txt")
    metadata = output / "provenance"
    metadata.mkdir()
    for path in (ROOT / "configs").glob("*.lock.json"):
        shutil.copyfile(path, metadata / path.name)
    for rel in ["src/music_cover/review.py", "src/music_cover/pipeline.py", "scripts/stage_runner.py"]:
        shutil.copyfile(ROOT / rel, metadata / Path(rel).name)
    stems = json.loads((output / "stems.json").read_text())
    midi = json.loads((output / "midi.json").read_text())
    descriptive = {name: describe_audio(request[name]) for name in ("original", "final")}
    save_json(output / "audio_comparison.json", descriptive)

    # Fixed-offset excerpts allow fast A/B review without pretending source and cover are beat-aligned.
    excerpt_dir = output / "excerpts"
    excerpt_dir.mkdir()
    n = stems["duration_s"]
    starts = sorted(set([min(30.0, max(0.0, n - 20)), max(0.0, n / 2 - 10), max(0.0, n - 40)]))
    excerpt_index = []
    for index, start in enumerate(starts, 1):
        for role, source in [("final", request["final"]), ("lead_vocals", output / "audio/lead_vocals.wav"),
                             ("backing_vocals", output / "audio/backing_vocals.wav"), ("instrumental", output / "audio/instrumental.wav")]:
            path = excerpt_dir / f"{index:02d}_{role}.wav"
            with sf.SoundFile(source) as stream:
                stream.seek(round(start * stream.samplerate))
                data = stream.read(round(20 * stream.samplerate), dtype="float32", always_2d=True)
                sf.write(path, data, stream.samplerate, subtype="FLOAT")
            excerpt_index.append({"file": str(path.relative_to(output)), "start_s": start, "duration_s": len(data) / RATE})
    save_json(output / "excerpts.json", excerpt_index)

    with (output / "review_scores.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["reviewer", "asset", "criterion", "start_seconds", "end_seconds", "score_1_to_5", "confidence", "comments"])
        for criterion in ["原旋律保留", "曲风差异", "音色相似度", "演唱自然度", "歌词清晰与完整度", "音准", "节奏", "混音平衡", "伪影与杂音"]:
            writer.writerow(["", "../final.wav", criterion, "", "", "", "", ""])
        for role in stems["stems"]:
            for criterion in ["目标声部保留", "其他声部串音", "瞬态与混响损伤"]:
                writer.writerow(["", f"audio/{role}.wav", criterion, "", "", "", "", ""])
        for role, spec in midi["tracks"].items():
            for criterion in ["音高准确性", "起止时刻准确性", "漏音与多余音符"]:
                writer.writerow(["", spec["file"], criterion, "", "", "", "", ""])

    report = ("专业评审数据包\n\n"
              "所有 review/audio 轨道均来自换声后的最终混音，48 kHz、立体声、32-bit float，同起点同长度。\n"
              "主唱/和声与各乐器是模型分离估计，可能有串音；背景人声残差可包含混响，不能据此断言有独立和声演唱。\n"
              "vocals 和 instrumental 是父轨；重建请只相加 stems.json 的 leaf_sum_roles，包含残差。\n"
              "重建误差小仅证明交付完整，不证明分离准确。残差轨不是独立乐器。\n\n"
              "MIDI 来自最终音频及分轨的重新识别。harmony.mid 是和弦指南，不是实际伴奏的完整演奏谱。\n"
              "drums_onsets.mid 只是低/中/高频起音候选，GM 36/38/42 是便于试听的映射，需要人工修正。\n"
              "MIDI 保持音频绝对秒时间，120 BPM 仅为时间刻度，不是歌曲速度；未量化。\n"
              "原始模型 MIDI（含可用的弯音）在 raw_midi，音符 CSV 在 notes，模型激活在 activations。\n"
              "final_transcription 含成品的和弦、调性、拍点、段落估计；source_transcription 仅为输入原曲对照。\n"
              "MIDI 无人声音色；不能代替音频评判孙燕姿音色。\n\n"
              "先对比 original.wav、restyled.flac、final.wav 与 reference.wav，再逐轨听音并核对 MIDI。\n"
              "lyrics_input.txt 保存本次输入歌词，用于评判漏词、错词和演唱顺序。\n"
              "excerpts 是固定时间窗口，不保证对应主歌/副歌；原曲与成品也不能按相同秒数视为音乐对齐。\n"
              "review_scores.csv 可由多位评审填写：1=很差，3=一般，5=很好；记录时间戳、置信度及意见。\n"
              "自动检查覆盖文件完整、采样点对齐、重建与 MIDI 合法性；主观听感及转谱正确率未自动验收。\n")
    (output / "README.txt").write_text(report)
    players = "".join(f'<h3>{html.escape(LABELS[role])}</h3><audio controls preload="none" src="audio/{role}.wav"></audio>'
                      f'<a href="audio/{role}.wav" download>下载 WAV</a>' for role in stems["stems"])
    rows = "".join(f'<tr><td>{html.escape(role)}</td><td>{spec["notes"]}</td><td>{html.escape(spec["status"])}</td>'
                   f'<td><a href="{spec["file"]}" download>MIDI</a></td></tr>' for role, spec in midi["tracks"].items())
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>专业评审数据</title>'
            '<style>body{max-width:960px;margin:32px auto;padding:0 20px;font:16px/1.6 sans-serif;color:#222;background:#fafafa}'
            'audio{display:block;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccc}pre{white-space:pre-wrap}</style>'
            '<h1>成品分轨与 MIDI 评审</h1><p><a href="../listen.html">完整歌曲与原曲对照</a> · '
            '<a href="review_scores.csv" download>人工评分表</a> · <a href="midi_bundle.zip" download>MIDI 数据包</a></p>'
            '<details open><summary>评审说明与限制</summary><pre>' + html.escape(report) + '</pre></details>'
            '<h2>MIDI 与音符数据</h2><table><tr><th>声部</th><th>音符数</th><th>状态</th><th>下载</th></tr>' + rows + '</table>'
            '<h2>分轨试听</h2>' + players + '</html>')
    (output / "index.html").write_text(page)
    with zipfile.ZipFile(output / "midi_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ["midi", "notes", "raw_midi", "final_transcription", "source_transcription"]:
            for path in sorted((output / directory).rglob("*")):
                if path.is_file() and path.suffix in {".mid", ".csv", ".lab", ".abc", ".tsv", ".json"}:
                    archive.write(path, path.relative_to(output))
        for name in ["README.txt", "midi.json", "review_scores.csv", "audio_comparison.json"]:
            archive.write(output / name, name)
        if (output / "lyrics_input.txt").is_file():
            archive.write(output / "lyrics_input.txt", "lyrics_input.txt")
    save_json(output / "manifest.json", {
        "schema_version": 1, "subjective_quality": "awaiting_human_review",
        "files": {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in sorted(output.rglob("*")) if p.is_file() and p.name not in
                  {"manifest.json", "process.log", "command.json"}}})


def run_stage(args):
    request = json.loads(Path(args.request).read_text())
    {"review_stems": separate_review, "review_midi": transcribe_stems,
     "review_package": build_review_package}[args.stage](request, Path(args.output))
