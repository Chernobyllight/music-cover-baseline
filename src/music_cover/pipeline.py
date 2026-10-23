from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .audio import decode, inspect_audio, mix_tracks, sha256
from .pitch import ABC_HELPER, choose_transposition, transpose_abc
from .review import artifact_paths

ROOT = Path(__file__).resolve().parents[2]
STAGE_SCRIPT = ROOT / "scripts/stage_runner.py"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def clean_lyrics(text):
    text = text.lstrip("\ufeff")
    rows, timed = [], []
    timestamp = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d+))?\]")
    for index, line in enumerate(text.splitlines()):
        tags = list(timestamp.finditer(line))
        line = timestamp.sub("", line).strip()
        if re.match(r"^\[(?:ar|ti|al|by|offset|length):", line, re.I):
            continue
        if re.match(r"^(?:作词|作曲|编曲|制作人|混音|母带|词|曲)\s*[:：]", line):
            continue
        rows.append(line)
        if line:
            for tag in tags:
                fraction = float("0." + tag[3]) if tag[3] else 0.0
                timed.append((int(tag[1]) * 60 + int(tag[2]) + fraction, index, line))
    if timed:
        rows = [line for _, _, line in sorted(timed)]
    cleaned = "\n".join(rows).strip()
    if not cleaned or not re.search(r"[\w\u4e00-\u9fff]", cleaned):
        raise ValueError("Lyrics are empty after removing metadata")
    if not re.search(r"\[(?:Verse|Chorus|Intro|Bridge|Outro|Pre-Chorus|Interlude)\]", cleaned, re.I):
        cleaned = "[Verse]\n" + cleaned
    return cleaned + "\n"


class Pipeline:
    def __init__(self, run, request):
        self.run, self.request = Path(run), request
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run / ".lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.run / "deliverables").exists():
            history = self.run / "deliverables_history"
            history.mkdir(exist_ok=True)
            (self.run / "deliverables").rename(history / (dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]))
        self.state_path = self.run / "stage_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"stages": {}}
        self.state["status"] = "running"
        self.state["request"] = request
        write_json(self.run / "request.json", request)
        write_json(self.state_path, self.state)
        self.environment = os.environ.copy()
        self.environment.update({
            "CUDA_VISIBLE_DEVICES": str(request["gpu"]), "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
            "HF_HOME": str(ROOT / "cache/huggingface"),
            "HF_HUB_DOWNLOAD_TIMEOUT": "180", "HF_HUB_ETAG_TIMEOUT": "60",
            "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
            "USE_TF": "0", "USE_FLAX": "0",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        })

    def execute(self, env, stage, output, *arguments):
        command = [str(ROOT / f"envs/{env}/bin/python"), str(STAGE_SCRIPT), stage,
                   "--output", str(output), *map(str, arguments)]
        write_json(output / "command.json", {"argv": command, "gpu": self.request["gpu"], "cwd": str(ROOT)})
        with (output / "process.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, env=self.environment,
                                     stdout=log, stderr=subprocess.STDOUT, timeout=10800)
        if process.returncode:
            tail = (output / "process.log").read_text(errors="replace")[-4500:]
            raise RuntimeError(f"{stage} failed ({process.returncode}); log: {output / 'process.log'}\n{tail}")

    def stage(self, name, inputs, params, action):
        fingerprint = identity({"inputs": {key: sha256(value) for key, value in inputs.items()},
                                "params": params, "adapter_sha256": sha256(STAGE_SCRIPT),
                                "audio_sha256": sha256(Path(__file__).with_name("audio.py")),
                                "pipeline_sha256": sha256(Path(__file__)),
                                "pitch_sha256": sha256(Path(__file__).with_name("pitch.py")),
                                "abc_parser_sha256": sha256(ABC_HELPER),
                                "review_sha256": sha256(Path(__file__).with_name("review.py"))
                                if "review" in name else None})
        previous = self.state["stages"].get(name, {})
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "succeeded":
            outputs = previous.get("outputs", {})
            if outputs and all(Path(item["path"]).is_file() and sha256(item["path"]) == item["sha256"]
                               for item in outputs.values()):
                print(f"[reuse] {name}", flush=True)
                return {key: Path(item["path"]) for key, item in outputs.items()}
        base = self.run / name
        base.mkdir(exist_ok=True)
        number = len(list(base.glob("attempt_*"))) + 1
        directory = base / f"attempt_{number:03d}"
        directory.mkdir()
        start = time.monotonic()
        record = {"status": "running", "fingerprint": fingerprint, "attempt": str(directory),
                  "params": params, "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        self.state["stages"][name] = record
        write_json(self.state_path, self.state)
        print(f"[run] {name} -> {directory}", flush=True)
        try:
            paths = action(directory)
            output_records = {}
            for role, path in paths.items():
                path = Path(path)
                if not path.is_file() or not path.stat().st_size:
                    raise RuntimeError(f"Missing stage artifact: {path}")
                item = {"path": str(path), "sha256": sha256(path)}
                if path.suffix.lower() in {".wav", ".flac", ".mp3"}:
                    item["audio"] = inspect_audio(path)
                output_records[role] = item
            record.update(status="succeeded", outputs=output_records, wall_seconds=time.monotonic() - start)
        except BaseException as error:
            record.update(status="failed", error=str(error), wall_seconds=time.monotonic() - start)
            self.state["status"] = "failed"
            write_json(directory / "failure.json", record)
            delivery = self.run / "deliverables"
            delivery.mkdir(exist_ok=True)
            (delivery / "report.md").write_text(f"本次流程未完成，失败阶段：{name}。\n\n{error}\n\n日志目录：{directory}\n")
            raise
        finally:
            write_json(self.state_path, self.state)
        write_json(directory / "result.json", record)
        print(f"[done] {name}: {record['wall_seconds']:.1f}s", flush=True)
        return {key: Path(value["path"]) for key, value in output_records.items()}

    def run_all(self):
        r = self.request
        model_lock = ROOT / "configs/generation_models.lock.json"
        model_identity = sha256(model_lock) if model_lock.exists() else "download-incomplete"

        def prepare(out):
            source, reference = out / "source.wav", out / "reference.wav"
            source_meta = decode(r["song"], source)
            ref_meta = decode(r["reference"], reference, rate=44100, channels=1,
                              start=r["reference_start"], duration=r["reference_duration"])
            if ref_meta["duration_s"] < 1.0 or ref_meta["rms"] < 1e-5:
                raise ValueError("Reference must contain at least one second of audible speech or singing")
            (out / "lyrics.txt").write_text(clean_lyrics(Path(r["lyrics"]).read_text(encoding="utf-8-sig")))
            write_json(out / "metadata.json", {"source": source_meta, "reference": ref_meta,
                       "reference_note": "Reference audio is cropped but not automatically source-separated"})
            return {"source": source, "reference": reference, "lyrics": out / "lyrics.txt"}

        prepared = self.stage("00_prepare", {key: r[key] for key in ("song", "lyrics", "reference")},
                              {"start": r["reference_start"], "duration": r["reference_duration"]}, prepare)

        def transcribe(out):
            self.execute("sheetsage2", "transcribe", out, "--input", prepared["source"])
            score = out / "score.abc"
            if "V:" not in score.read_text():
                raise ValueError("Transcription did not produce a voiced ABC score")
            return {"score": score, "events": out / "events.json",
                    **{f"artifact:{k}": v for k, v in artifact_paths(out).items()}}

        transcription = self.stage("01_transcribe", {"source": prepared["source"]},
                                   {"models": model_identity, "mode": "melody"}, transcribe)

        adaptation = None
        generation_score = transcription["score"]
        if str(r.get("transpose", "0")) != "0":
            def source_separate(out):
                self.execute("separator", "separate", out, "--input", prepared["source"])
                return {"vocals": out / "vocals.wav", "instrumental": out / "instrumental.wav"}

            source_stems = self.stage("01a_source_separate", {"source": prepared["source"]},
                                      {"model": "bs_roformer_ep_317", "sample_rate": 44100}, source_separate)

            def adapt(out):
                write_json(out / "pitch_request.json", {"tracks": {
                    "source": str(source_stems["vocals"]), "reference": str(prepared["reference"])}})
                self.execute("seed-vc", "pitch", out, "--request", out / "pitch_request.json")
                profiles = json.loads((out / "pitch_profiles.json").read_text())["tracks"]
                plan = choose_transposition(profiles["source"]["whole"], profiles["reference"]["whole"], r["transpose"])
                revised, invariants = transpose_abc(transcription["score"].read_text(), plan["selected_semitones"])
                plan["score_checks"] = invariants
                plan["seed_vc_semitones"] = 0
                (out / "score_transposed.abc").write_text(revised)
                shutil.copyfile(transcription["score"], out / "score_original.abc")
                write_json(out / "transposition.json", plan)
                print(f"[pitch] selected {plan['selected_semitones']:+d} semitones; "
                      f"{invariants['source_key']} -> {invariants['target_key']}", flush=True)
                return {"score": out / "score_transposed.abc", "original_score": out / "score_original.abc",
                        "plan": out / "transposition.json", "profiles": out / "pitch_profiles.json",
                        "source_f0": out / "source_f0.npz", "reference_f0": out / "reference_f0.npz"}

            adaptation = self.stage("01b_pitch_adapt", {"source_vocals": source_stems["vocals"],
                                     "reference": prepared["reference"], "score": transcription["score"]},
                                    {"transpose": r["transpose"], "rmvpe": sha256(ROOT / "models/RMVPE/rmvpe.pt")}, adapt)
            generation_score = adaptation["score"]

        def generate(out):
            request = {"style": r["style"], "lyrics": prepared["lyrics"].read_text(),
                       "seed": r["seed"], "cot": "melody"}
            write_json(out / "generation_request.json", request)
            self.execute("yue2", "cover", out, "--request", out / "generation_request.json",
                         "--score", generation_score)
            return {"cover": out / "generated/audio.flac"}

        generated = self.stage("02_cover", {"score": generation_score, "lyrics": prepared["lyrics"]},
                               {"models": model_identity, "style": r["style"], "seed": r["seed"]}, generate)

        def separate(out):
            decode(generated["cover"], out / "separation_input.wav")
            self.execute("separator", "separate", out, "--input", out / "separation_input.wav")
            return {"vocals": out / "vocals.wav", "instrumental": out / "instrumental.wav"}

        separated = self.stage("03_separate", generated, {"model": "bs_roformer_ep_317", "sample_rate": 44100,
                                                         "normalization_threshold": 1.0}, separate)

        def convert(out):
            decode(separated["vocals"], out / "source_mono.wav", rate=44100, channels=1)
            self.execute("seed-vc", "convert", out, "--input", out / "source_mono.wav",
                         "--reference", prepared["reference"], "--steps", r["steps"], "--seed", r["seed"])
            return {"converted": out / "converted.wav"}

        converted = self.stage("04_convert", {"vocals": separated["vocals"], "reference": prepared["reference"]},
                               {"model": "seed_vc_f0_44k_ema_v2", "steps": r["steps"], "f0": True,
                                "auto_f0": False, "length_adjust": 1.0, "semitones": 0,
                                "seed": r["seed"], "assets": sha256(ROOT / "configs/voice_models.lock.json")}, convert)

        def mix(out):
            result = mix_tracks(generated["cover"], separated["vocals"], separated["instrumental"],
                                converted["converted"], out)
            write_json(out / "mix.json", result)
            return {"final": out / "final.wav", "final_mp3": out / "final.mp3",
                    "reconstruction": out / "reconstruction.wav",
                    "aligned_vocal": out / "converted_aligned.wav", "metrics": out / "mix.json"}

        mixed = self.stage("05_mix", {**generated, **separated, **converted},
                           {"output_rate": 48000, "mp3_bitrate_kbps": 192}, mix)
        pitch_check = None
        if adaptation:
            def check_pitch(out):
                write_json(out / "pitch_request.json", {"tracks": {
                    "before": str(separated["vocals"]), "after": str(mixed["aligned_vocal"])}})
                self.execute("seed-vc", "pitch", out, "--request", out / "pitch_request.json")
                return {"profiles": out / "pitch_profiles.json", "before_f0": out / "before_f0.npz",
                        "after_f0": out / "after_f0.npz"}
            pitch_check = self.stage("06_pitch_check", {"before": separated["vocals"], "after": mixed["aligned_vocal"]},
                                      {"rmvpe": sha256(ROOT / "models/RMVPE/rmvpe.pt")}, check_pitch)
        review = None
        if r.get("review", True):
            review_models = sha256(ROOT / "configs/review_models.lock.json")

            def review_stems(out):
                write_json(out / "stems_request.json", {
                    "final": str(mixed["final"]), "vocal": str(mixed["aligned_vocal"]),
                    "instrumental": str(separated["instrumental"]), "mix_metrics": str(mixed["metrics"]), "seed": r["seed"]})
                self.execute("separator", "review_stems", out, "--request", out / "stems_request.json")
                return artifact_paths(out)

            stems = self.stage("07_review_stems", {"final": mixed["final"], "vocal": mixed["aligned_vocal"],
                               "instrumental": separated["instrumental"], "metrics": mixed["metrics"]},
                               {"models": review_models, "seed": r["seed"], "output_rate": 48000}, review_stems)

            def transcribe_final(out):
                self.execute("sheetsage2", "transcribe_final", out, "--input", mixed["final"])
                return artifact_paths(out)

            final_score = self.stage("08_final_transcribe", {"final": mixed["final"]},
                                     {"models": model_identity, "melody_only": False}, transcribe_final)

            def review_midi(out):
                write_json(out / "midi_request.json", {"stems": str(stems["stems.json"].parent),
                           "transcription": str(final_score["transcription_summary.json"].parent)})
                self.execute("separator", "review_midi", out, "--request", out / "midi_request.json")
                return artifact_paths(out)

            midi = self.stage("09_review_midi", {"stems": stems["stems.json"], "chords": final_score["chords.mid"]},
                              {"models": review_models, "timebase_bpm": 120}, review_midi)

            def review_package(out):
                write_json(out / "package_request.json", {
                    "stems": str(stems["stems.json"].parent),
                    "transcription": str(final_score["transcription_summary.json"].parent),
                    "midi": str(midi["midi.json"].parent), "source_transcription": str(transcription["score"].parent),
                    "original": str(prepared["source"]), "final": str(mixed["final"]), "run_request": r})
                self.execute("separator", "review_package", out, "--request", out / "package_request.json")
                return artifact_paths(out)

            review = self.stage("10_review_package", {"stems": stems["stems.json"], "midi": midi["midi.json"],
                                "final_score": final_score["transcription_summary.json"],
                                "source_events": transcription["events"], "original": prepared["source"], "final": mixed["final"]},
                                {"schema_version": 1, "models": review_models}, review_package)
        self.deliver(prepared, generated, separated, converted, mixed, adaptation, pitch_check, review)
        self.state["status"] = "completed"
        self.state["subjective_quality"] = "requires_listening"
        write_json(self.state_path, self.state)
        write_json(self.run / "manifest.json", self.state)
        print(f"\nRESULTS: {self.run / 'deliverables'}", flush=True)

    def deliver(self, prepared, generated, separated, converted, mixed, adaptation=None, pitch_check=None, review=None):
        destination = self.run / "deliverables"
        destination.mkdir(exist_ok=True)
        items = {"original.wav": prepared["source"], "reference.wav": prepared["reference"],
                 "restyled.flac": generated["cover"], "reconstruction.wav": mixed["reconstruction"],
                 "final.wav": mixed["final"], "final.mp3": mixed["final_mp3"],
                 "vocals_before.wav": separated["vocals"],
                 "vocals_after.wav": mixed["aligned_vocal"], "instrumental.wav": separated["instrumental"]}
        if adaptation:
            items.update({"score_original.abc": adaptation["original_score"], "score_transposed.abc": adaptation["score"],
                          "transposition.json": adaptation["plan"], "pitch_input.json": adaptation["profiles"],
                          "pitch_output.json": pitch_check["profiles"]})
        if review:
            items.update({"review/" + name: path for name, path in review.items()})
        for name, source in items.items():
            temporary = destination / (name + ".tmp")
            temporary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, temporary)
            temporary.replace(destination / name)
        metrics = json.loads(mixed["metrics"].read_text())
        write_json(destination / "checks.json", metrics)
        duration = metrics["final"]["duration_s"]
        times = "\n".join(f"| {name} | {stage['wall_seconds']:.1f} |"
                          for name, stage in self.state["stages"].items() if stage["status"] == "succeeded")
        report = (
            f"完整流程已执行，输出 {duration:.2f} 秒、48 kHz 立体声。\n\n"
            "自动文件、数值、时长与峰值检查已执行；音乐风格、音色相似度和自然度仍需试听验收。\n\n"
            f"目标风格：{self.request['style']}\n\n"
            "依次试听 original.wav、restyled.flac、final.wav；参考人物声音在 reference.wav。"
            "final.mp3 是同一最终混音的 192 kbps 压缩版，便于下载和分享。"
            "reconstruction.wav 用于检查分离和重混的损伤。\n\n"
            f"转换人声长度差：{metrics['alignment']['converted_duration_delta_s']:.6f} 秒。"
            f"固定延迟校正：{metrics['alignment']['applied_shift_samples']} 个 48 kHz 采样点。\n\n"
            f"最终峰值估计：{metrics['final']['estimated_true_peak_dbtp']:.2f} dBTP"
            "（4 倍过采样工程估计）。\n\n"
            f"对齐需要重点试听：{metrics['alignment']['needs_listening_review']}。\n\n"
            "| 阶段 | 耗时秒，包含进程启动和模型加载 |\n| --- | ---: |\n" + times + "\n\n"
            "完整参数、输入哈希和阶段日志保存在上一级目录。\n"
        )
        if Path(self.request["song"]).name == "tonight-awake.mp3":
            report += ("\n本次演示使用 YuE2 官方发布的 AI 生成中文歌曲及其配套歌词，"
                       "目标音色使用公开 LibriSpeech 语音样本；不是用户目标人物的录音。"
                       "来源与哈希见项目 data/demo_sources.json。\n")
        if adaptation:
            plan = json.loads(adaptation["plan"].read_text())
            checks = plan["score_checks"]
            report += (f"\n本次谱面整体移调 {plan['selected_semitones']:+d} 个半音："
                       f"{checks['source_key']} → {checks['target_key']}。人声及器乐音符均已移调，"
                       "音符起点、时值、节拍和速度通过逐事件检查。Seed-VC 不再额外移调。\n\n"
                       "参考音区来自本次短片段，不代表歌手完整音域；自动选调为启发式建议。"
                       "选择依据见 transposition.json、pitch_input.json，生成和换声后实测见 pitch_output.json。"
                       "谱面检查不保证生成歌声逐音遵循；仍需检查开头是否改善以及高音是否自然。\n")
        if review:
            report += ("\n成品分轨和 MIDI 评审包已生成，入口 review/index.html。\n"
                       "所有 review/audio 轨道均同起点、同采样率、同长度；父轨不能与子轨重复相加。\n"
                       "MIDI 来自成品重新转谱，自动估计仍需专业人员校对；鼓为起音候选，和弦谱为分析指南。\n"
                       "音符 CSV、原始模型数据、校验哈希、响度/节奏描述和人工评分表均包含在 review。\n")
        (destination / "report.md").write_text(report, encoding="utf-8")
        labels = [("original.wav", "1 原曲"), ("reference.wav", "目标音色参考"),
                  ("restyled.flac", "2 换风格"), ("final.wav", "3 最终混音（WAV）"),
                  ("final.mp3", "最终混音（MP3，192 kbps）"),
                  ("reconstruction.wav", "分离后直接重混对照"),
                  ("vocals_before.wav", "换声前人声"), ("vocals_after.wav", "换声后人声")]
        players = "\n".join(f'<p>{html.escape(label)} · <a href="{name}" download>下载</a></p>'
                            f'<audio controls preload="none" src="{name}"></audio>'
                            for name, label in labels)
        page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>音乐 baseline 试听</title>'
                '<style>body{max-width:850px;margin:40px auto;padding:0 20px;font:17px/1.6 sans-serif;background:#f7f7f4;color:#222}'
                'audio{width:100%}p{margin-top:28px}pre{white-space:pre-wrap;font:inherit}</style>'
                '<h1>音乐 baseline 试听</h1><p>完整流程已执行；听感质量请对照以下音频验收。</p>'
                + ('<p><a href="review/index.html">打开成品分轨与 MIDI 专业评审包</a></p>' if review else '')
                + players + '<details><summary>运行报告</summary><pre>' + html.escape(report) + '</pre></details></html>')
        (destination / "listen.html").write_text(page, encoding="utf-8")


def preflight(review=True):
    required = [ROOT / f"envs/{name}/bin/python" for name in ["control", "sheetsage2", "yue2", "separator", "seed-vc"]]
    required += [ROOT / f"models/{name}/model.safetensors" for name in ["SheetSage2", "MERT-v2-FullSong", "YuE2-3B", "YuE2-Vae"]]
    required += [ROOT / "third_party/seed-vc/inference.py", ROOT / "configs/generation_models.lock.json"]
    required += [ROOT / "configs/voice_models.lock.json", ROOT / "configs/seed_svc.local.yml"]
    if review:
        required += [ROOT / "configs/review_models.lock.json", ROOT / "models/review/basic_pitch.onnx",
                     ROOT / "models/review/demucs/htdemucs_6s.yaml", ROOT / "models/review/demucs/5c90dfd2-34c22ccb.th",
                     ROOT / "models/separation/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
                     ROOT / "models/separation/config_mel_band_roformer_karaoke.yaml"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Setup incomplete:\n" + "\n".join(missing))
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not available")
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              check=True, capture_output=True, text=True, timeout=30)
    if "libmp3lame" not in encoders.stdout:
        raise RuntimeError("ffmpeg must support libmp3lame to export final.mp3")
    if review:
        lock = json.loads((ROOT / "configs/review_models.lock.json").read_text())
        for entry in lock["assets"]:
            path = ROOT / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise RuntimeError(f"Review model missing or changed: {path}")
        subprocess.run([str(ROOT / "envs/separator/bin/python"), "-c",
                        "from demucs.pretrained import get_model; from basic_pitch.inference import predict; import mido, pretty_midi"],
                       check=True, capture_output=True, text=True, timeout=60)


def main():
    parser = argparse.ArgumentParser(description="原曲 → 换风格 → 分离 → 参考音色换声 → 混音")
    parser.add_argument("--demo", action="store_true", help="使用已下载的完整中文演示歌曲")
    parser.add_argument("--resume", type=Path, help="恢复指定运行目录，可同时替换参考音频等参数")
    for name in ["song", "lyrics", "reference", "style", "name"]:
        parser.add_argument("--" + name)
    parser.add_argument("--gpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--reference-start", type=float)
    parser.add_argument("--reference-duration", type=float)
    parser.add_argument("--transpose", help="auto：按参考歌声及原曲音区选升调量；或 -12..12 个半音，默认 0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--review", action=argparse.BooleanOptionalAction, default=None,
                        help="默认生成成品分轨、MIDI、专业评审数据；--no-review 仅输出基础结果")
    args = parser.parse_args()
    if args.demo and args.resume:
        parser.error("--demo and --resume cannot be combined")
    if args.preflight:
        preflight(review=args.review is not False)
        print("Preflight files OK")
        return
    request = {"name": "cover", "seed": 42, "steps": 30, "gpu": "0", "reference_start": 0.0, "reference_duration": 15.0,
               "transpose": "0", "review": True}
    if args.demo:
        request.update(json.loads((ROOT / "configs/demo.json").read_text()))
    if args.resume:
        request.update(json.loads((args.resume.resolve() / "request.json").read_text()))
    for name in ["song", "lyrics", "reference", "style", "name", "seed", "steps", "reference_start", "reference_duration", "transpose", "review"]:
        if getattr(args, name) is not None:
            request[name] = getattr(args, name)
    if args.gpu is not None:
        request["gpu"] = args.gpu
    if request["transpose"] != "auto":
        try:
            value = int(request["transpose"])
            if not -12 <= value <= 12:
                raise ValueError()
            request["transpose"] = str(value)
        except (ValueError, TypeError):
            parser.error("transpose must be auto or an integer between -12 and 12")
    for key in ["song", "lyrics", "reference", "style"]:
        if not request.get(key):
            parser.error(f"--{key} is required (or use --demo)")
    if not math.isfinite(request["reference_start"]) or not math.isfinite(request["reference_duration"]) or request["reference_start"] < 0 or not 1 <= request["reference_duration"] <= 25:
        parser.error("reference start must be >= 0; duration must be between 1 and 25 seconds")
    if not 1 <= request["steps"] <= 200:
        parser.error("steps must be between 1 and 200")
    if not 0 <= request["seed"] < 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")
    for key in ["song", "lyrics", "reference"]:
        path = Path(request[key]).expanduser()
        request[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
        if not Path(request[key]).is_file():
            parser.error(f"File does not exist: {request[key]}")
    preflight(review=request["review"])
    name = re.sub(r"[^\w-]+", "_", request["name"])[:60]
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + name + "_" + uuid.uuid4().hex[:6]
    run = args.resume.resolve() if args.resume else ROOT / "runs" / run_id
    pipeline = Pipeline(run, request)
    print(f"RUN: {run}", flush=True)
    try:
        pipeline.run_all()
    finally:
        pipeline.lock.close()


if __name__ == "__main__":
    main()
