"""Create a fresh Seed-VC training YAML from verified separated recordings."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

import soundfile as sf
import yaml

from .audio import sha256
from .common.paths import ROOT, read_run_state
from .common.voice_paths import portable_path

BASE_CHECKPOINT = "models/Seed-VC/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
BASE_CONFIG = "models/Seed-VC/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
from .common.voice_config import (
    SEPARATION_TOOL, read_json, stage_artifacts, validation_windows, write_new_config,
    checked_recordings as _checked_recordings,
)


def path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def portable(value):
    return portable_path(path(value), ROOT)



def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voice-id", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--recordings-dir", required=True, help="Completed voice_data/separate output directory")
    p.add_argument("--source-run", required=True, help="Song run with successful stages 00 and 04")
    p.add_argument("--validation-ids", nargs="+", required=True)
    p.add_argument("--test-ids", nargs="+", required=True)
    p.add_argument("--validation-windows", nargs="+", required=True, metavar="START:DURATION", help="Seconds in stage 04 audio")
    p.add_argument("--output", help="Default: configs/voice_training/<voice-id>_seedvc_<version>.yaml")
    p.add_argument("--dataset-dir", help="Default: data/voice_training/<voice-id>/<version>/dataset")
    p.add_argument("--export-dir", help="Default: models/voices/<voice-id>/seed_vc/<version>")
    p.add_argument("--base-checkpoint", default=BASE_CHECKPOINT)
    p.add_argument("--base-config", default=BASE_CONFIG)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--segment-seconds", type=float, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-steps", type=int, default=30)
    p.add_argument("--validation-seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--plan", action="store_true", help="Validate and print YAML without writing files or loading models")
    return p


def check_parameters(args):
    for name in ("voice_id", "version"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')} must be a simple ID, without spaces or path separators")
    for name in ("max_steps", "save_every", "batch_size", "validation_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.learning_rate <= 1e-3 or not 1 <= args.segment_seconds <= 30:
        raise ValueError("learning-rate must be in (0, 0.001]; segment-seconds must be in [1, 30]")
    if args.gpu < 0 or not all(0 <= x < 2**32 for x in (args.seed, args.validation_seed)):
        raise ValueError("gpu must be nonnegative; seeds must be in [0, 2**32)")
    for name in ("validation_ids", "test_ids"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate --{name.replace('_', '-')} values")
    if set(args.validation_ids) & set(args.test_ids):
        raise ValueError("validation-ids and test-ids overlap")


def checked_recordings(args):
    return _checked_recordings(args, ROOT)




def build_config(args):
    """Read-only validation and construction; no model loading or output creation."""
    check_parameters(args)
    output = path(args.output or f"configs/voice_training/{args.voice_id}_seedvc_{args.version}.yaml")
    dataset = path(args.dataset_dir or f"data/voice_training/{args.voice_id}/{args.version}/dataset")
    export = path(args.export_dir or f"models/voices/{args.voice_id}/seed_vc/{args.version}")
    targets = {"output": output, "dataset-dir": dataset, "export-dir": export}
    for name, file in targets.items():
        if file.exists() or file.is_symlink():
            raise ValueError(f"--{name} already exists; choose a fresh version/path: {file}")
    for name, file in targets.items():
        for other, location in targets.items():
            if other != name and (file.is_relative_to(location) or location.is_relative_to(file)):
                raise ValueError("output, dataset-dir and export-dir must not contain one another")
    rows, excluded, manifest_file = checked_recordings(args)
    state = read_run_state(path(args.source_run), root=ROOT)
    reference = stage_artifacts(state, "00_prepare", ["reference"])["reference"]
    stems = stage_artifacts(state, "04_vocal_separation", ["vocals", "instrumental"])
    reference_sha = sha256(reference)
    if reference_sha in {r["sha256"] for r in rows}:
        raise ValueError("Reference prompt must be held out from train/validation/test recordings")
    info = {role: sf.info(str(file)) for role, file in stems.items()}
    if abs(info["vocals"].duration - info["instrumental"].duration) > 0.25:
        raise ValueError("Stage 04 vocals/instrumental durations differ by more than 0.25 seconds")
    windows = validation_windows(args.validation_windows, min(x.duration for x in info.values()))
    checkpoint, model_config = path(args.base_checkpoint), path(args.base_config)
    for file in (checkpoint, model_config):
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"Base model asset missing/empty: {file}")
    model = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    if not isinstance(model, dict) or not isinstance(model.get("model_params"), dict):
        raise ValueError("base-config must be a Seed-VC model architecture YAML, not an old training task YAML")
    cfg = {
        "voice_id": args.voice_id, "version": args.version, "gpu": args.gpu,
        "smoke_test": False, "segment_seconds": args.segment_seconds,
        "dataset_dir": portable(dataset),
        "sources": [{"path": r["path"], "group": f"{args.voice_id}:{r['id']}", "split": r["split"]} for r in rows],
        "reference": portable(reference), "base_checkpoint": portable(checkpoint), "base_config": portable(model_config),
        "training": {"seed": args.seed, "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                     "max_steps": args.max_steps, "save_every": args.save_every},
        "validation": {"source_run": portable(path(args.source_run)), "windows": windows,
                       "steps": args.validation_steps, "seed": args.validation_seed},
        "export_dir": portable(export),
        "configuration_generation": {
            "tool": "music_cover.voice_training_config", "schema_version": 1,
            "separation_manifest": portable(manifest_file), "separation_manifest_sha256": sha256(manifest_file),
            "excluded_ids": excluded,
            "checked_stage_artifacts": {role: {"path": portable(file), "sha256": sha256(file)}
                                        for role, file in {"reference": reference, **stems}.items()},
        },
    }
    summary = {"python": sys.executable, "output": str(output),
               "recordings": dict(Counter(r["split"] for r in rows)), "excluded_ids": excluded,
               "seconds": {split: round(sum(r["duration_s"] for r in rows if r["split"] == split), 3)
                           for split in ("train", "validation", "test")},
               "source_duration_s": info["vocals"].duration}
    return cfg, output, summary



def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        config, output, summary = build_config(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.plan:
            print("PLAN: checks passed; no files created.\n" + yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        else:
            write_new_config(output, config)
            entry = ROOT / "scripts/05_voice_conversion/finetuned/training/00_prepare/run.sh"
            print(f"Created training config: {output}\nNext stage script: {entry}")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
