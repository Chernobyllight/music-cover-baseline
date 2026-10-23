#!/usr/bin/env python3
"""Fetch a complete public YuE2 example song, its lyrics and a speech reference."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download

    revision = "8f24312f187763b9854adeec87e437e2b03bbff1"
    song = ROOT / "data/raw/songs/tonight-awake.mp3"
    reference = ROOT / "data/raw/references/librispeech-reference.flac"
    lyrics = ROOT / "data/raw/lyrics/tonight-awake.txt"
    meta = ROOT / "data/raw/lyrics/tonight-awake.source.json"
    for name, target in [("assets/audio/tonight-awake.mp3", song),
                         ("examples/tonight-awake.json", meta)]:
        source = hf_hub_download("m-a-p/YuE2-3B", name, revision=revision)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if digest(song) != "54f642933f796e4244f0a308d3d0603c09b4d87f04e0bc42fbcbb9df0bffbcf9":
        raise RuntimeError("Demo source song checksum mismatch")
    lyrics.write_text(json.loads(meta.read_text())["lyrics"] + "\n", encoding="utf-8")
    ref_revision = HfApi().dataset_info("Narsil/asr_dummy").sha
    source = hf_hub_download("Narsil/asr_dummy", "1.flac", repo_type="dataset", revision=ref_revision)
    reference.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, reference)
    provenance = {
        "song": {"title": "今晚不眠", "type": "public AI-generated example, not a human studio recording",
                 "url": "https://huggingface.co/m-a-p/YuE2-3B", "revision": revision,
                 "path": str(song.relative_to(ROOT)), "sha256": digest(song)},
        "lyrics": {"path": str(lyrics.relative_to(ROOT)), "sha256": digest(lyrics),
                   "source": "examples/tonight-awake.json from the same pinned repository"},
        "reference": {"url": "https://huggingface.co/datasets/Narsil/asr_dummy",
                      "revision": ref_revision, "file": "1.flac",
                      "path": str(reference.relative_to(ROOT)), "sha256": digest(reference),
                      "purpose": "public speech reference for zero-shot pipeline verification"},
    }
    (ROOT / "data/demo_sources.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    request = {
        "song": str(song.relative_to(ROOT)), "lyrics": str(lyrics.relative_to(ROOT)),
        "reference": str(reference.relative_to(ROOT)),
        "style": "Mandarin Chinese, acoustic folk ballad, solo lead vocal, warm acoustic guitar, soft piano, light brushed drums, intimate clear dry vocal, restrained arrangement",
        "seed": 42, "steps": 30, "reference_start": 0.0, "reference_duration": 15.0,
        "name": "tonight_awake_folk",
    }
    (ROOT / "configs/demo.json").write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
