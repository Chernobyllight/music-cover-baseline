#!/usr/bin/env python3
"""Record the environment actually used, without modifying shared packages."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def output(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def main():
    record = {"captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "ffmpeg": output(["ffmpeg", "-version"]).splitlines()[0],
              "gpu": output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]),
              "environments": {}, "upstream": {}}
    for name in ["control", "sheetsage2", "yue2", "separator", "seed-vc"]:
        python = str(ROOT / "envs" / name / "bin/python")
        code = """import sys,torch,numpy,importlib.metadata as m,json
packages = {}
for p in ['torchvision','torchaudio','transformers','huggingface-hub','safetensors','accelerate','soundfile','scipy','librosa','numba','onnxruntime','audio-separator','triton','cuda-bindings']:
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
        (ROOT / "configs" / f"{name}.installed.txt").write_text(freeze + "\n")
    for name in ["YuE", "seed-vc"]:
        record["upstream"][name] = output(["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party" / name)
    (ROOT / "configs/environments.lock.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
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
    (ROOT / "configs/local_assets.lock.json").write_text(json.dumps(assets, indent=2) + "\n")
    print("Saved configs/environments.lock.json and configs/local_assets.lock.json")


if __name__ == "__main__":
    main()
