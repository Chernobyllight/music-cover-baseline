#!/usr/bin/env python3
"""Download/copy review models atomically, then record versions and SHA256.

Run with envs/separator/bin/python after installing configs/review-requirements.txt.
--local-assets supports a workstation download when server networking is slow.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KARAOKE = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
ASSETS = [
    ("models/separation/" + KARAOKE, "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/" + KARAOKE),
    ("models/separation/config_mel_band_roformer_karaoke.yaml", "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/config_mel_band_roformer_karaoke.yaml"),
    ("models/review/demucs/5c90dfd2-34c22ccb.th", "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th"),
    ("models/review/demucs/htdemucs_6s.yaml", "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/htdemucs_6s.yaml"),
]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-assets", type=Path)
    args = parser.parse_args()
    records = []
    for relative, url in ASSETS:
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            local = args.local_assets / path.name if args.local_assets else None
            try:
                if local is not None and local.is_file():
                    shutil.copyfile(local, temporary)
                else:
                    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                        shutil.copyfileobj(response, stream)
                if temporary.stat().st_size == 0:
                    raise RuntimeError(f"Empty download: {url}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        checksum = digest(path)
        if path.name.endswith("34c22ccb.th") and not checksum.startswith("34c22ccb"):
            raise RuntimeError("Demucs checkpoint hash does not match upstream filename")
        records.append({"path": relative, "url": url, "bytes": path.stat().st_size, "sha256": checksum})
    from basic_pitch import __file__ as package_file
    source = Path(package_file).parent / "saved_models/icassp_2022/nmp.onnx"
    target = ROOT / "models/review/basic_pitch.onnx"
    shutil.copyfile(source, target)
    records.append({"path": str(target.relative_to(ROOT)), "url": "https://github.com/spotify/basic-pitch",
                    "source": "basic-pitch 0.4.0 wheel bundled ONNX model", "bytes": target.stat().st_size,
                    "sha256": digest(target)})
    packages = {name: importlib.metadata.version(name) for name in [
        "audio-separator", "demucs", "basic-pitch", "onnxruntime", "numpy", "torch", "torchaudio",
        "librosa", "mido", "pretty-midi", "dora-search", "julius", "openunmix", "lameenc", "mir_eval"]}
    lock = {"assets": records, "packages": packages,
            "hash_note": "SHA256 measured after download; only Demucs filename publishes a short hash.",
            "runtime_note": "Reuse existing Torch 2.8 CUDA runtime; Basic Pitch uses ONNX CPU, no TensorFlow."}
    (ROOT / "configs/review_models.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
