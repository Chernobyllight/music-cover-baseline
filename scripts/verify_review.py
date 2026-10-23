#!/usr/bin/env python3
"""Independent delivered-artifact checks; no model inference and no quality score."""
import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import zipfile

import mido
import numpy as np
import pretty_midi
import soundfile as sf


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attributes):
        self.targets += [v for k, v in attributes if k in {"href", "src"} and v]


def verify(run):
    deliver = run / "deliverables"
    root = deliver / "review"
    manifest = json.loads((root / "manifest.json").read_text())
    for rel, item in manifest["files"].items():
        path = (root / rel).resolve()
        assert path.is_relative_to(root.resolve()), rel
        assert path.is_file() and path.stat().st_size == item["bytes"], rel
        assert digest(path) == item["sha256"], f"Hash mismatch: {rel}"
    stems = json.loads((root / "stems.json").read_text())
    assert digest(deliver / "final.wav") == stems["input_final_sha256"]
    expected = {"lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano", "other",
                "instrumental_vocal_leak", "instrumental_residual", "mix_residual"}
    assert set(stems["leaf_sum_roles"]) == expected
    for role, spec in stems["stems"].items():
        path = root / spec["path"]
        info = sf.info(path)
        assert (info.samplerate, info.channels, info.frames) == (48000, 2, stems["frames"]), role
        assert info.subtype == "FLOAT", role
        assert digest(path) == spec["sha256"], role
    readers = [sf.SoundFile(root / stems["stems"][role]["path"]) for role in stems["leaf_sum_roles"]]
    maximum, count, squared_error = 0.0, 0, 0.0
    try:
        with sf.SoundFile(deliver / "final.wav") as final:
            while True:
                reference = final.read(65536, dtype="float64", always_2d=True)
                if not len(reference):
                    break
                parts = [stream.read(len(reference), dtype="float64", always_2d=True) for stream in readers]
                assert all(x.shape == reference.shape and np.isfinite(x).all() for x in parts)
                error = np.sum(parts, axis=0) - reference
                maximum = max(maximum, float(np.max(np.abs(error))))
                squared_error += float(np.sum(error ** 2))
                count += error.size
        assert maximum < 2e-6, maximum
    finally:
        for stream in readers:
            stream.close()
    midi_summary = json.loads((root / "midi.json").read_text())
    midi_counts = {}
    for role, spec in midi_summary["tracks"].items():
        path = root / spec["file"]
        parsed = pretty_midi.PrettyMIDI(str(path))
        notes = [n for instrument in parsed.instruments for n in instrument.notes]
        assert len(notes) == spec["notes"], role
        assert abs(mido.MidiFile(path).length - stems["duration_s"]) < 0.002, role
        assert all(0 <= n.start < n.end <= stems["duration_s"] + 0.002 for n in notes), role
        if role == "drums_onsets":
            assert all(i.is_drum for i in parsed.instruments), role
        midi_counts[role] = len(notes)
    assert midi_counts["lead_vocals"] > 0, "Lead MIDI unexpectedly empty"
    assert midi_counts["arrangement_estimate"] > 0
    for page in [deliver / "listen.html", root / "index.html"]:
        parser = Links()
        parser.feed(page.read_text())
        for target in parser.targets:
            url = urlsplit(target)
            if not url.scheme and url.path:
                path = (page.parent / unquote(url.path)).resolve()
                assert path.is_relative_to(deliver.resolve()) and path.is_file(), str(path)
    with zipfile.ZipFile(root / "midi_bundle.zip") as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert (root / name).resolve().is_relative_to(root.resolve())
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest(root / name), name
    state = json.loads((run / "manifest.json").read_text())
    assert state["status"] == "completed"
    assert all(state["stages"][name]["status"] == "succeeded" for name in [
        "07_review_stems", "08_final_transcribe", "09_review_midi", "10_review_package"])
    return {"technical_validation": "passed", "checked_payload_files": len(manifest["files"]),
            "duration_s": stems["duration_s"], "audio_stems": len(stems["stems"]),
            "leaf_stems": len(expected), "reconstruction_max_abs_error": maximum,
            "reconstruction_rms_error": (squared_error / count) ** 0.5,
            "midi_note_counts": midi_counts,
            "empty_midi_tracks": [role for role, count in midi_counts.items() if not count],
            "subjective_quality": "awaiting_human_review", "request": state["request"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run.resolve())
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
