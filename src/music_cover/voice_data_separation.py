"""Prepare target-singer recordings independently of song generation/training."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid

from .audio import inspect_audio, sha256
from .common.io import identity, write_json
from .common.paths import ROOT

TOOL = "music_cover.voice_data_separation"
VOCALS_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
LEAD_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aiff"}
ROLES = {"vocals": ("vocals", "instrumental"),
         "lead": ("vocals", "instrumental", "lead", "backing")}


def project_path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def portable(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir")
    p.add_argument("--pattern", default="*/original.m4a", help="Glob relative to input-dir")
    p.add_argument("--output-dir")
    p.add_argument("--exclude-ids", nargs="*", default=[])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--mode", choices=ROLES, default="lead")
    p.add_argument("--model-dir", default="models/separation")
    p.add_argument("--vocals-model", default=VOCALS_MODEL)
    p.add_argument("--lead-model", default=LEAD_MODEL)
    p.add_argument("--lead-overlap", type=int, default=4)
    p.add_argument("--timeout-seconds", type=int, default=10800, help="Per recording")
    p.add_argument("--plan", action="store_true", help="Check and list inputs; write nothing, load no models")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true", help="Reuse only verified matching results")
    action.add_argument("--force", action="store_true", help="Recompute; archive existing selected results")
    p.add_argument("--worker-request", help=argparse.SUPPRESS)
    return p


def discover(args):
    source, output = project_path(args.input_dir), project_path(args.output_dir)
    if not source.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Input and output directories must not contain one another")
    if Path(args.pattern).is_absolute() or ".." in Path(args.pattern).parts:
        raise ValueError("--pattern must stay inside input-dir")
    rows, identifiers = [], set()
    for file in sorted(source.glob(args.pattern)):
        if not file.is_file() or file.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        if not file.resolve().is_relative_to(source):
            raise ValueError(f"Input symlink escapes input-dir: {file}")
        # The library uses <recording-id>/original.m4a; flat folders use file stems.
        key = file.stem if file.parent == source else file.parent.name
        if key.startswith(".") or key in identifiers:
            raise ValueError(f"Ambiguous/reserved recording id {key!r}; narrow --pattern or use unique recording folders")
        identifiers.add(key)
        rows.append({"id": key, "source": portable(file), "source_sha256": sha256(file),
                     "excluded": key in args.exclude_ids})
    unknown = set(args.exclude_ids) - identifiers
    if unknown:
        raise ValueError(f"Excluded ids do not match any input: {sorted(unknown)}")
    if not any(not r["excluded"] for r in rows):
        raise ValueError("No selected audio files; check input-dir/pattern/exclusions")
    return source, output, rows


def preflight(args):
    if args.gpu < 0 or not 2 <= args.lead_overlap <= 50 or args.timeout_seconds <= 0:
        raise ValueError("gpu >= 0, lead-overlap 2..50 and positive timeout-seconds are required")
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise ValueError(f"{name} is not available in base PATH")
    if importlib.util.find_spec("audio_separator.separator") is None:
        raise ValueError("audio-separator is not installed in the current base interpreter")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA is unavailable for --gpu {args.gpu} in {sys.executable}")
    models = project_path(args.model_dir)
    names = [args.vocals_model] + ([args.lead_model] if args.mode == "lead" else [])
    assets = {}
    for name in names:
        if Path(name).name != name:
            raise ValueError("Model names must be filenames; set their directory with --model-dir")
        file = models / name
        if not file.is_file() or not file.stat().st_size:
            raise ValueError(f"Model weight missing/empty: {file}")
        assets[name] = sha256(file)
    # audio-separator discovers model configuration filenames from its catalog.
    # Include local configs so a configuration edit can never reuse stale audio.
    configs = sorted([*models.glob("*.yaml"), *models.glob("*.yml")])
    if not configs:
        raise ValueError(f"No local separator model YAML configurations in {models}")
    assets.update({p.name: sha256(p) for p in configs})
    return {"model_assets": assets, "implementation_sha256": sha256(__file__),
            "audio_separator_version": importlib.metadata.version("audio-separator"),
            "torch_version": torch.__version__, "mode": args.mode, "sample_rate": 44100,
            "normalization": 1.0, "amplification": 0.0,
            "vocals_model": args.vocals_model, "lead_model": args.lead_model if args.mode == "lead" else None,
            "vocals_overlap": 8, "lead_overlap": args.lead_overlap, "batch_size": 1}


def fingerprint(row, settings):
    # Content and processing parameters determine reuse, not a server mount name.
    return identity({"source_sha256": row["source_sha256"], "settings": settings})


def cached(folder, expected, mode):
    try:
        report = json.loads((folder / "separation.json").read_text())
        if report["tool"] != TOOL or report["status"] != "succeeded" or report["identity"] != expected:
            return False
        for role in ROLES[mode]:
            file = folder / f"{role}.wav"
            if file.is_symlink() or sha256(file) != report["outputs"][role]["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def check_output(output, rows, args):
    selected = {r["id"] for r in rows if not r["excluded"]}
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError(f"Output is not a directory: {output}")
    manifest = output / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get("tool") != TOOL or not isinstance(old.get("recordings"), list):
            raise ValueError("Unrecognized manifest in output-dir; use a new output-dir")
        old_selected = {r["id"] for r in old["recordings"] if not r["excluded"]}
        if old_selected != selected:
            raise ValueError("Output belongs to another recording selection; use a new output-dir")
    allowed = selected | {"manifest.json", "process.log", ".lock", ".work", ".history"}
    for child in output.iterdir():
        if child.is_symlink() or child.name not in allowed:
            raise ValueError(f"Unmanaged file/directory in output: {child}; use a new output-dir")
    if not (args.resume or args.force) and any((output / key).exists() for key in selected):
        raise ValueError("Recording outputs already exist; choose --resume or --force")


def execute_worker(request, directory, log):
    file = directory / "worker_request.json"
    write_json(file, request)
    argv = [sys.executable, "-m", TOOL, "--worker-request", str(file)]
    write_json(directory / "command.json", {"argv": argv, "python": sys.executable,
               "prefix": sys.prefix, "gpu": request["gpu"], "cwd": str(directory)})
    env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(request["gpu"]))
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(argv, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                   check=True, timeout=request["timeout_seconds"])


def run_batch(args, output, rows, settings, runner=execute_worker):
    check_output(output, rows, args)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_output(output, rows, args)
        report = {"tool": TOOL, "schema_version": 1, "status": "running", "path_base": "project",
                  "input_dir": portable(project_path(args.input_dir)), "output_dir": portable(output),
                  "pattern": args.pattern, "settings": settings,
                  "runtime": {"python": sys.executable, "prefix": sys.prefix, "gpu": args.gpu},
                  "recordings": [{**r, "status": "excluded" if r["excluded"] else "pending"} for r in rows]}
        def save():
            report["counts"] = dict(Counter(r["status"] for r in report["recordings"]))
            write_json(output / "manifest.json", report)
        save()
        with (output / "process.log").open("a", buffering=1) as batch_log:
            def message(text):
                print(text, flush=True)
                batch_log.write(text + "\n")
            message(f"[{datetime.now(timezone.utc).isoformat()}] python={sys.executable}; gpu={args.gpu}")
            try:
                for row in report["recordings"]:
                    if row["excluded"]:
                        continue
                    folder = output / row["id"]
                    key = fingerprint(row, settings)
                    if args.resume and cached(folder, key, args.mode):
                        row.update(status="skipped", output=portable(folder), identity=key)
                        message(f"[verified/skip] {row['id']}")
                        save()
                        continue
                    # Never overwrite an unverified successful result during resume.
                    if folder.exists() and not args.force:
                        row.update(status="failed", error="Existing output failed cache verification; use --force to archive/recompute")
                        message(f"[failed] {row['id']}: {row['error']}")
                        save()
                        continue
                    (output / ".work").mkdir(exist_ok=True)
                    work = Path(tempfile.mkdtemp(prefix=row["id"] + "-", dir=output / ".work"))
                    row.update(status="running", work_dir=portable(work), identity=key)
                    save()
                    message(f"[start] {row['id']}; log={work / 'process.log'}")
                    request = {"source": str(project_path(row["source"])), "settings": settings,
                               "model_dir": str(project_path(args.model_dir)), "gpu": args.gpu,
                               "timeout_seconds": args.timeout_seconds}
                    try:
                        with (work / "process.log").open("w") as log:
                            runner(request, work, log)
                        outputs = verify_outputs(work, project_path(row["source"]), args.mode)
                        if sha256(project_path(row["source"])) != row["source_sha256"]:
                            raise ValueError("Input changed during separation")
                        write_json(work / "separation.json", {
                            "tool": TOOL, "status": "succeeded", "identity": key,
                            "source": row["source"], "source_sha256": row["source_sha256"],
                            "settings": settings, "runtime": report["runtime"], "outputs": outputs,
                            "quality_status": "technical_checks_only; human_listening_required"})
                        backup = None
                        if folder.exists():
                            backup = output / ".history" / (row["id"] + "-" + uuid.uuid4().hex)
                            backup.parent.mkdir(exist_ok=True)
                            folder.rename(backup)
                        try:
                            work.rename(folder)
                        except BaseException:
                            if backup is not None:
                                backup.rename(folder)
                            raise
                        row.pop("work_dir", None)
                        row.update(status="succeeded", output=portable(folder))
                        message(f"[ok] {row['id']}: {folder}")
                    except Exception as error:
                        row.update(status="failed", error=str(error))
                        write_json(work / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
                        message(f"[failed] {row['id']}: {error}; inspect {work / 'process.log'}")
                    save()
            except BaseException:
                report["status"] = "interrupted"
                save()
                raise
            report["status"] = "failed" if any(r["status"] == "failed" for r in report["recordings"]) else "succeeded"
            save()
            message(f"[{report['status']}] {report['counts']}; manifest={output / 'manifest.json'}")
    return 0 if report["status"] == "succeeded" else 1


def verify_outputs(directory, source, mode):
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "json", str(source)], check=True, capture_output=True, text=True, timeout=60)
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid source duration")
    outputs = {}
    for role in ROLES[mode]:
        file = directory / f"{role}.wav"
        info = inspect_audio(file)
        if info["sample_rate"] != 44100 or abs(info["duration_s"] - duration) > .25:
            raise ValueError(f"{role}: sample rate or duration differs from source")
        if role in {"vocals", "lead"} and info["rms"] < 1e-5:
            raise ValueError(f"{role}: effectively silent; do not use as singing training data")
        outputs[role] = {"path": file.name, **info}
    return outputs


def worker(filename):
    # One isolated process per recording releases model memory even on failure.
    request = json.loads(Path(filename).read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(request["gpu"])
    from audio_separator.separator import Separator
    import torch
    torch.set_num_threads(4)
    directory = Path(filename).resolve().parent
    os.chdir(directory)  # audio-separator 0.30.2 soundfile writes relative to cwd.
    settings = request["settings"]
    stages = [("vocals", request["source"], settings["vocals_model"], settings["vocals_overlap"],
               {"Vocals": "vocals", "Instrumental": "instrumental"})]
    if settings["mode"] == "lead":
        stages.append(("lead", str(directory / "vocals.wav"), settings["lead_model"],
                       settings["lead_overlap"], {"Vocals": "lead", "Instrumental": "backing"}))
    for stage, source, model, overlap, names in stages:
        write_json(directory / "progress.json", {"stage": stage, "status": "running"})
        print(f"[{stage}] model={model}; input={source}", flush=True)
        separator = Separator(output_dir=str(directory), output_format="WAV",
            model_file_dir=request["model_dir"], normalization_threshold=1.0,
            amplification_threshold=0.0, sample_rate=44100, use_soundfile=True,
            mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                         "batch_size": 1, "overlap": overlap, "pitch_shift": 0})
        separator.load_model(model_filename=model)
        separator.separate(source, names)
        del separator
        gc.collect()
        torch.cuda.empty_cache()
    write_json(directory / "progress.json", {"stage": stages[-1][0], "status": "succeeded"})


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.worker_request:
        worker(args.worker_request)
        return 0
    if not args.input_dir or not args.output_dir:
        p.error("--input-dir and --output-dir are required")
    try:
        source, output, rows = discover(args)
        check_output(output, rows, args)
        settings = preflight(args)
        print(f"Input: {source}\nOutput: {output}\nPython: {sys.executable}", flush=True)
        for row in rows:
            print(f"{'exclude' if row['excluded'] else 'process'}: {row['id']} <- {row['source']}", flush=True)
        if args.plan:
            print(json.dumps({"selected": sum(not r["excluded"] for r in rows),
                              "excluded": sum(r["excluded"] for r in rows), "mode": args.mode,
                              "model_assets": settings["model_assets"], "writes": False}, ensure_ascii=False, indent=2))
            return 0
        return run_batch(args, output, rows, settings)
    except (OSError, ValueError, ImportError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
