#!/usr/bin/env python3
"""Download the pinned transcription and generation models into this project."""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

MODELS = {
    "YuE2-3B": ("m-a-p/YuE2-3B", "8f24312f187763b9854adeec87e437e2b03bbff1"),
    "YuE2-Vae": ("m-a-p/YuE2-Vae", "152733a19ad43aa67e367f9b5503ef8075bb5126"),
    "SheetSage2": ("m-a-p/SheetSage2", "b83c3d79a6d98bcffaf8f9ed112c90aa887887bc"),
    "MERT-v2-FullSong": ("m-a-p/MERT-v2-FullSong", "7001f7657276238629e61c7746ce70735e0604c0"),
}


def download(item):
    from huggingface_hub import snapshot_download

    name, (repo, revision) = item
    target = ROOT / "models" / name
    print(f"Downloading {repo}@{revision} -> {target}", flush=True)
    snapshot_download(
        repo_id=repo, revision=revision, local_dir=target, max_workers=4,
        allow_patterns=["*.py", "*.json", "*.safetensors", "*.tiktoken",
                        "requirements*.txt", "README.md", "LICENSE*",
                        "THIRD_PARTY*", "licenses/*", "render_assets/*"],
    )
    print(f"READY {name}", flush=True)
    return name, {"repo_id": repo, "revision": revision, "path": str(target)}


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = dict(pool.map(download, MODELS.items()))
    path = ROOT / "configs/generation_models.lock.json"
    path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
