#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

SPECS = {
    "Seed-VC": ("Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e",
                ["DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"]),
    "BigVGAN": ("nvidia/bigvgan_v2_44khz_128band_512x", "95a9d1dcb12906c03edd938d77b9333d6ded7dfb",
                ["config.json", "bigvgan_generator.pt", "LICENSE"]),
    "Whisper-small": ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d",
                      ["*.json", "*.txt", "model.safetensors"]),
    "RMVPE": ("lj1995/VoiceConversionWebUI", "e6d0c1a17da07c33557852f9dfa2bd44cc75737d", ["rmvpe.pt"]),
    "CAMPPlus": ("funasr/campplus", "e4b6ede7ce16997aff4ae69fbca1f0175e2afede", ["campplus_cn_common.bin"]),
}


def download(item):
    from huggingface_hub import snapshot_download
    name, (repo, revision, patterns) = item
    path = ROOT / "models" / name
    print(f"Downloading {repo}@{revision}", flush=True)
    snapshot_download(repo, revision=revision, local_dir=path, allow_patterns=patterns, max_workers=3)
    print(f"READY {name}", flush=True)
    return name, {"repo_id": repo, "revision": revision, "path": str(path)}


def main():
    import yaml
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        records = dict(pool.map(download, SPECS.items()))
    config = yaml.safe_load((ROOT / "models/Seed-VC/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml").read_text())
    config["model_params"]["vocoder"]["name"] = str(ROOT / "models/BigVGAN")
    config["model_params"]["speech_tokenizer"]["name"] = str(ROOT / "models/Whisper-small")
    (ROOT / "configs/seed_svc.local.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    (ROOT / "configs/voice_models.lock.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
