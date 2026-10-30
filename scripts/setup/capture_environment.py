#!/usr/bin/env python3
"""Record the environment actually used, without modifying shared packages."""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def output(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'configs',
                        help='Use a cache directory for a read-only runtime snapshot; default updates configs locks')
    parser.add_argument('--skip-assets', action='store_true', help='Skip full model-directory hashing')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {"captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "ffmpeg": output(["ffmpeg", "-version"]).splitlines()[0],
              "gpu": output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]),
              "environments": {}, "upstream": {}}
    for name in ["control", "sheetsage2", "yue2", "separator", "seed-vc", "base"]:
        python = "/opt/conda/bin/python" if name == "base" else str(ROOT / "envs" / name / "bin/python")
        code = """import sys,torch,numpy,importlib.metadata as m,json
packages = {}
for p in ['torchvision','torchaudio','transformers','huggingface-hub','safetensors','accelerate','soundfile','scipy','librosa','numba','onnxruntime','audio-separator','triton','cuda-bindings','mido','pretty-midi','basic-pitch','demucs','speechbrain','faiss-cpu']:
    try: packages[p] = m.version(p)
    except m.PackageNotFoundError: pass
for distribution in m.distributions():
    if distribution.metadata['Name'].startswith('nvidia-'):
        packages.setdefault(distribution.metadata['Name'], distribution.version)
print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':numpy.__version__,'packages':packages}))
"""
        versions = json.loads(output([python, "-c", code]))
        record["environments"][name] = versions
        freeze = output([python, "-m", "pip", "freeze", "--local"])
        (args.output_dir / f"{name}.installed.txt").write_text(freeze + "\n")
    for name in ["YuE", "seed-vc", "so-vits-svc", "rvc"]:
        record["upstream"][name] = output(["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party" / name)
    (args.output_dir / "environments.lock.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    if args.skip_assets:
        print(f"Saved environment snapshot to {args.output_dir}; model hashes skipped")
        return
    # Compute hashes locally; distinguish them from the upstream download revision locks.
    assets = {}
    for path in sorted((ROOT / "models").rglob("*")):
        if not path.is_file() or any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(ROOT / "models").parts):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        assets[str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    (args.output_dir / "local_assets.lock.json").write_text(json.dumps(assets, indent=2) + "\n")
    print(f"Saved environment and asset snapshots to {args.output_dir}")


if __name__ == "__main__":
    main()
