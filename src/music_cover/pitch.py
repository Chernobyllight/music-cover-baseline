"""Pitch profiling, conservative register selection, and verified native ABC edits."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ABC_HELPER = ROOT / "third_party/YuE/skills/yue2-music/scripts/abc_tools.py"


def abc_tools():
    name = "_music_cover_native_abc"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ABC_HELPER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def pitch_class(name):
    return (dict(zip("CDEFGAB", (0, 2, 4, 5, 7, 9, 11)))[name[0]]
            + name.count("#") - name.count("b")) % 12


def shifted_key(key, semitones):
    tools = abc_tools()
    minor = key.endswith("m")
    pc = (pitch_class(key.removesuffix("m")) + semitones) % 12
    choices = [k for k in tools.KEYS if k.endswith("m") == minor
               and pitch_class(k.removesuffix("m")) == pc]
    return min(choices, key=lambda k: (abs(tools.KEYS[k]), tools.KEYS[k] > 0))


def transpose_abc(text, semitones):
    """Transpose interpreted pitches, keeping durations, ties, voices and bar grids.

    Parse the source first, then consume its *sounding* events while rewriting
    individual tied tokens. This avoids guessing accidental/tie scope from text.
    Accidentals are emitted only when required by the destination key/bar.
    """
    if not isinstance(semitones, int) or not -12 <= semitones <= 12:
        raise ValueError("Transpose must be an integer between -12 and 12")
    tools = abc_tools()
    before = tools.parse(text)
    lines = text.splitlines()
    key = shifted_key(lines[7][2:], semitones)
    keys = {v: key for v in tools.VOICES}
    positions = dict.fromkeys(tools.VOICES, -1)
    tied = dict.fromkeys(tools.VOICES, False)
    current_voice = None

    def spell(pitch, key_name):
        if not 0 <= pitch <= 127:
            raise ValueError(f"Transposition exceeds MIDI range: {pitch}")
        names = ("C Db D Eb E F Gb G Ab A Bb B" if tools.KEYS[key_name] < 0
                 else "C C# D D# E F F# G G# A A# B").split()
        name = names[pitch % 12]
        octave = pitch // 12 - 1
        letter = name[0].lower() if octave >= 5 else name[0]
        marks = "'" * max(0, octave - 5) + "," * max(0, 4 - octave)
        return letter, marks, name.count("#") - name.count("b")

    for index, line in enumerate(lines):
        if line in ("V: Vocal", "V: Ins"):
            current_voice = line[3:]
        if line.startswith("K:"):
            new_key = shifted_key(line[2:], semitones)
            lines[index] = "K:" + new_key
            if current_voice:
                keys[current_voice] = new_key
        if index not in before.music_lines:
            continue
        voice = before.music_lines[index]
        rewritten = []
        for bar in line[:-1].split("|"):
            local = {}

            def change(match):
                chord, inline_key = match.group("chord", "key")
                if chord is not None:
                    def root(m):
                        letter, _, alteration = spell(60 + (pitch_class(m[0]) + semitones) % 12, keys[voice])
                        return letter.upper() + ("b" if alteration < 0 else "#" if alteration > 0 else "")
                    return '"' + re.sub(r"^[A-G](?:bb|##|b|#)?|(?<=/)[A-G](?:bb|##|b|#)?", root, chord) + '"'
                if inline_key is not None:
                    keys[voice] = shifted_key(inline_key, semitones)
                    local.clear()
                    return "[K:" + keys[voice] + "]"
                if match.group("note") == "z":
                    return match[0]
                if not tied[voice]:
                    positions[voice] += 1
                pitch = before.voices[voice].notes[positions[voice]][1] + semitones
                letter, marks, alteration = spell(pitch, keys[voice])
                active = local.get(letter.upper(), tools.key_accidentals(keys[voice])[letter.upper()])
                accidental = ""
                if alteration != active:
                    accidental = {-1: "_", 0: "=", 1: "^"}[alteration]
                    local[letter.upper()] = alteration
                tied[voice] = bool(match.group("tie"))
                return accidental + letter + marks + match.group("duration") + match.group("tie")

            rewritten.append(tools.TOKEN.sub(change, bar))
        lines[index] = "|".join(rewritten) + "|"
    output = "\n".join(lines) + "\n"
    after = tools.parse(output)
    if before.bpm != after.bpm or before.unit != after.unit:
        raise ValueError("Transposition changed tempo or rhythmic unit")
    counts = {}
    for name in tools.VOICES:
        a, b = before.voices[name], after.voices[name]
        expected_notes = [[t, p + semitones, d] for t, p, d in a.notes]
        expected_keys = [(t, shifted_key(k, semitones)) for t, k in a.keys]
        if a.bars != b.bars or expected_notes != b.notes or b.keys != expected_keys:
            raise ValueError(f"{name}: transposition violated note/time/key invariants")
        if len(a.chords) != len(b.chords):
            raise ValueError("Transposition changed chord count")
        for (ta, ca), (tb, cb) in zip(a.chords, b.chords):
            def parts(chord):
                return re.fullmatch(r"([A-G](?:bb|##|b|#)?)([^/]*)(?:/([A-G](?:bb|##|b|#)?))?", chord).groups()
            ra, qa, ba = parts(ca)
            rb, qb, bb = parts(cb)
            if (ta != tb or qa != qb or (pitch_class(rb) - pitch_class(ra)) % 12 != semitones % 12
                    or bool(ba) != bool(bb) or (ba and (pitch_class(bb) - pitch_class(ba)) % 12 != semitones % 12)):
                raise ValueError("Transposition violated chord invariants")
        counts[name] = len(b.notes)
    return output, {"semitones": semitones, "source_key": text.splitlines()[7][2:],
                    "target_key": key, "notes_checked": counts,
                    "pitches_transposed_exactly": True, "timing_and_meter_preserved": True,
                    "tempo_preserved": True, "chords_and_keys_transposed": True}


def summarize_pitch(f0, rms, start=0.0, end=None, hop_seconds=0.01):
    count = min(len(f0), len(rms))
    times = np.arange(count) * hop_seconds
    keep = (f0[:count] >= 65) & (f0[:count] <= 1000) & (rms[:count] >= 0.01)
    keep &= times >= start
    if end is not None:
        keep &= times < end
    values = np.asarray(f0[:count])[keep]
    result = {"voiced_seconds": float(len(values) * hop_seconds), "frames": int(len(values))}
    if not len(values):
        return result
    percentiles = [5, 10, 50, 90, 95]
    hz = np.percentile(values, percentiles)
    result["hz"] = {f"p{p}": float(v) for p, v in zip(percentiles, hz)}
    result["midi"] = {f"p{p}": float(69 + 12 * np.log2(v / 440)) for p, v in zip(percentiles, hz)}
    return result


def choose_transposition(source, reference, requested="auto"):
    for name, profile in [("source", source), ("reference", reference)]:
        if profile.get("voiced_seconds", 0) < 3 or "midi" not in profile:
            raise ValueError(f"Insufficient voiced singing for {name}: need >= 3 seconds")
    # The short reference is a sample, not the singer's measured maximum.
    # +5 st is only a declared exploratory headroom, not a physiological claim.
    target_low = reference["midi"]["p10"]
    tentative_high = reference["midi"]["p95"] + 5.0
    rows = []
    for shift in range(7):
        low = source["midi"]["p10"] + shift
        high = source["midi"]["p95"] + shift
        loss = (low - target_low) ** 2 + 4 * max(0.0, high - tentative_high) ** 2 + 0.10 * shift ** 2
        rows.append({"semitones": shift, "loss": loss, "predicted_low_midi": low,
                     "predicted_high_midi": high})
    selected = min(rows, key=lambda row: row["loss"])["semitones"] if requested == "auto" else int(requested)
    if not -12 <= selected <= 12:
        raise ValueError("Manual transposition must be between -12 and 12")
    return {"mode": requested, "selected_semitones": selected,
            "method": "match source P10 to reference P10, penalize P95 above sample P95+5st and large shifts",
            "auto_candidates": rows, "auto_search_semitones": [0, 6],
            "reference_sample_only": True, "exploratory_upper_headroom_semitones": 5,
            "warning": "Short reference does not establish singer range; audio adherence and quality require review."}
