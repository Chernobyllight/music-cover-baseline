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

def build_review_package(request, output):
    output = Path(output)
    for key, directory in [("stems", ""), ("transcription", "final_transcription"), ("midi", "")]:
        source = Path(request[key])
        for rel, path in artifact_paths(source).items():
            destination = output / directory / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    for name, source in request.get('pitch_adaptation', {}).items():
        destination = output / 'pitch_adaptation' / Path(name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    source_transcription = Path(request["source_transcription"])
    target = output / "source_transcription"
    target.mkdir()
    for path in source_transcription.iterdir():
        if path.suffix in {".mid", ".abc", ".lab"} and path.is_file() and path.stat().st_size:
            shutil.copyfile(path, target / path.name)
    save_json(output / "request.json", request["run_request"])
    lyric_source = request.get("lyrics", request["run_request"].get("lyrics"))
    if lyric_source:
        shutil.copyfile(lyric_source, output / "lyrics_input.txt")
    metadata = output / "provenance"
    metadata.mkdir()
    for path in (ROOT / "configs").glob("*.lock.json"):
        shutil.copyfile(path, metadata / path.name)
    for path in sorted((ROOT / "src/music_cover").rglob("*.py")):
        destination = metadata / "code" / path.relative_to(ROOT / "src")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    stems = json.loads((output / "stems.json").read_text())
    midi = json.loads((output / "midi.json").read_text())
    descriptive = json.loads(Path(request["evaluation"]).read_text())["audio_comparison"]
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
              "review_scores.csv 为模板；请在运行目录 reviews 中填写副本，多位评审可分别填写：1=很差，3=一般，5=很好；记录时间戳、置信度及意见。\n"
              "pitch_adaptation（如有）包含 02 的选调证据和移调 MIDI；此 MIDI 为生成输入旋律谱，不是成品重新转谱。\n"
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
            + ('<p><a href="pitch_adaptation/pitch_report.html">输入谱移调报告与 MIDI</a></p>' if (output / 'pitch_adaptation/pitch_report.html').is_file() else '')
            + '<details open><summary>评审说明与限制</summary><pre>' + html.escape(report) + '</pre></details>'
            '<h2>MIDI 与音符数据</h2><table><tr><th>声部</th><th>音符数</th><th>状态</th><th>下载</th></tr>' + rows + '</table>'
            '<h2>分轨试听</h2>' + players + '</html>')
    (output / "index.html").write_text(page)
    with zipfile.ZipFile(output / "midi_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ["midi", "notes", "raw_midi", "final_transcription", "source_transcription", "pitch_adaptation"]:
            for path in sorted((output / directory).rglob("*")):
                if path.is_file() and path.suffix in {".mid", ".csv", ".lab", ".abc", ".tsv", ".json", ".html"}:
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
