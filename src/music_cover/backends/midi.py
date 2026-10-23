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
    params = {"onset_threshold": 0.5, "frame_threshold": 0.3, "minimum_note_ms": 90, **request.get("parameters", {})}
    summaries, combined = {}, []
    for role, program in PITCHED_ROLES.items():
        print(f"[review MIDI] {role}", flush=True)
        audio_path = root / "audio" / f"{role}.wav"
        # Do not amplify silent or weak stems to manufacture notes.
        rms = inspect_audio(audio_path)["rms"]
        events, rows = [], []
        if rms >= 1e-5:
            activations, raw_midi, events = predict(
                audio_path, model, onset_threshold=params["onset_threshold"], frame_threshold=params["frame_threshold"],
                minimum_note_length=params["minimum_note_ms"], minimum_frequency=30 if role == "bass" else None,
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
        "parameters": {key: params[key] for key in ("onset_threshold", "frame_threshold", "minimum_note_ms")},
        "limitations": ["All MIDI is automatically estimated; not ground truth or the generator's native score.",
                        "MIDI has notes, not the singer's timbre. GM programs are audition hints only.",
                        "120 BPM is a fixed seconds-to-ticks timebase; see final_transcription/beat.lab for estimated beats.",
                        "Main MIDI omits pitch bends to avoid conflicting bends in polyphony; raw_midi retains model output.",
                        "Drum files are onset candidates with approximate frequency-band labels, requiring manual correction.",
                        "Empty or near-silent parts remain explicit rather than receiving invented notes."]})
