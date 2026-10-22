#!/usr/bin/env python3
"""Compare two completed runs; retain measured F0, independent pYIN and A/B clips.

Run with envs/seed-vc/bin/python. This is diagnostic evidence, not a listening test.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import soundfile as sf
    import librosa
    from music_cover.audio import sha256
    from music_cover.pitch import summarize_pitch
    baseline, candidate = args.baseline.resolve(), args.candidate.resolve()
    requests = [json.loads((run / "request.json").read_text()) for run in (baseline, candidate)]
    conditions = {key: requests[0][key] == requests[1][key] for key in ["style", "seed", "steps"]}
    conditions["same_prepared_reference"] = (sha256(baseline / "deliverables/reference.wav")
                                               == sha256(candidate / "deliverables/reference.wav"))
    conditions["same_prepared_song"] = (sha256(baseline / "deliverables/original.wav")
                                          == sha256(candidate / "deliverables/original.wav"))
    if not all(conditions.values()):
        raise ValueError(f"Comparison conditions differ; inspect before making a paired comparison: {conditions}")
    output = candidate / "deliverables/pitch_comparison"
    output.mkdir(exist_ok=False)
    tracks = {f"{version}_{phase}": str(run / f"deliverables/vocals_{phase}.wav")
              for version, run in [("old", baseline), ("new", candidate)] for phase in ["before", "after"]}
    (output / "request.json").write_text(json.dumps({"tracks": tracks}, indent=2))
    with (output / "process.log").open("w") as log:
        subprocess.run([sys.executable, str(ROOT / "scripts/stage_runner.py"), "pitch",
                        "--request", str(output / "request.json"), "--output", str(output)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    plan = json.loads((candidate / "deliverables/transposition.json").read_text())
    analysis = {"baseline": str(baseline), "candidate": str(candidate),
                "intended_semitones": plan["selected_semitones"],
                "paired_conditions": conditions,
                "inputs_sha256": {name: sha256(path) for name, path in tracks.items()},
                "scope": "F0 estimates and distribution comparisons; regenerated performances need not align note-for-note; no listening claim",
                "windows": {}, "pyin_40_56_seconds": {}}
    arrays = {name: np.load(output / f"{name}_f0.npz") for name in tracks}
    for label, start, end in [("whole", 0, None), ("opening_30_60", 30, 60),
                              ("verse_38_68", 38, 68), ("high_240_270", 240, 270)]:
        row = {name: summarize_pitch(a["f0"], a["rms"], start, end) for name, a in arrays.items()}
        row["median_shift_semitones"] = {}
        for phase in ["before", "after"]:
            old, new = row[f"old_{phase}"], row[f"new_{phase}"]
            if "hz" in old and "hz" in new:
                row["median_shift_semitones"][phase] = float(12 * np.log2(new["hz"]["p50"] / old["hz"]["p50"]))
        analysis["windows"][label] = row
    first, second = arrays["new_before"], arrays["new_after"]
    length = min(len(first["f0"]), len(second["f0"]))
    a, b = first["f0"][:length], second["f0"][:length]
    voiced = ((a >= 65) & (a <= 1000) & (b >= 65) & (b <= 1000)
              & (first["rms"][:length] >= .01) & (second["rms"][:length] >= .01))
    cents = 1200 * np.log2(b[voiced] / a[voiced])
    analysis["conversion_pitch_preservation"] = {
        "common_voiced_seconds": float(voiced.sum() * .01),
        "median_signed_cents": float(np.median(cents)) if len(cents) else None,
        "median_absolute_cents": float(np.median(np.abs(cents))) if len(cents) else None,
        "within_100_cents_fraction": float(np.mean(np.abs(cents) <= 100)) if len(cents) else None}
    for name, path in tracks.items():
        wave, _ = librosa.load(path, sr=16000, mono=True, offset=40, duration=16)
        f0, voiced, _ = librosa.pyin(wave, sr=16000, fmin=65, fmax=800, frame_length=2048, hop_length=160)
        rms = librosa.feature.rms(y=wave, frame_length=2048, hop_length=160)[0]
        f0 = np.where(voiced & np.isfinite(f0), f0, 0)
        analysis["pyin_40_56_seconds"][name] = summarize_pitch(f0, rms)
    (output / "comparison.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    players = []
    for label, start, end in [("第一段主歌，30–75 秒", 30, 75), ("高音段，235–275 秒", 235, 275)]:
        for version, run in [("old", baseline), ("new", candidate)]:
            source = run / "deliverables/final.wav"
            rate = sf.info(source).samplerate
            samples, rate = sf.read(source, start=start * rate, stop=end * rate, dtype="float32")
            name = f"{version}_{start}_{end}.wav"
            sf.write(output / name, samples, rate, subtype="PCM_24")
            players.append(f'<p>{"旧版" if version == "old" else "升调版"}：{label}</p><audio controls preload="none" src="{name}"></audio>')
    for version, run in [("旧版完整歌曲", baseline), ("升调版完整歌曲", candidate)]:
        link = os.path.relpath(run / "deliverables/final.wav", output)
        players.append(f'<p>{version}</p><audio controls preload="none" src="{html.escape(link)}"></audio>')
    summary = (f"本次谱面统一升 {plan['selected_semitones']} 个半音。对照使用相同曲风、随机种子和参考歌声。"
               "片段为直接截取，未额外调音高或归一化响度。请重点比较第一段主歌的低沉粗糙程度及高音段自然度。"
               "音高统计支持检查升调是否发生，不能判定音色相似度或听感改善。")
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>升调前后试听对照</title>'
            '<style>body{max-width:850px;margin:40px auto;padding:0 20px;font:17px/1.6 sans-serif}audio{width:100%}</style>'
            '<h1>《落叶归根》孙燕姿参考音色：升调对照</h1><p>' + summary + '</p>'
            + "\n".join(players) + '<p><a href="comparison.json">实测音高与独立复核</a> · <a href="../report.md">运行报告</a></p></html>')
    (output / "listen.html").write_text(page)
    print(json.dumps({"output": str(output), "opening": analysis["windows"]["opening_30_60"]["median_shift_semitones"],
                      "conversion": analysis["conversion_pitch_preservation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
