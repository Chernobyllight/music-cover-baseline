#!/usr/bin/env python3
"""Fetch the selected official UVR release asset with resumable range requests."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/" + NAME
SIZE = 639331213


def main():
    destination = ROOT / "models/separation" / NAME
    parts = ROOT / "cache/separator_parts"
    parts.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size != SIZE:
        # Resolve GitHub's redirect once, avoiding repeated API calls.
        with urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": "bytes=0-0"}), timeout=30) as response:
            resolved = response.url
        chunk_size = 4 * 1024**2
        def download(index):
            start, end = index * chunk_size, min(SIZE, (index + 1) * chunk_size) - 1
            part = parts / str(index)
            for attempt in range(40):
                downloaded = part.stat().st_size if part.exists() else 0
                if downloaded == end - start + 1:
                    return
                position = start + downloaded
                try:
                    request = urllib.request.Request(resolved, headers={"Range": f"bytes={position}-{end}"})
                    with urllib.request.urlopen(request, timeout=30) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {position}-{end}/{SIZE}":
                            raise RuntimeError("Unexpected HTTP range")
                        deadline = time.monotonic() + 15
                        with part.open("ab") as stream:
                            while block := response.read(64 * 1024):
                                stream.write(block)
                                if time.monotonic() >= deadline:
                                    break
                except Exception:
                    if attempt == 39:
                        raise
                    time.sleep(min(attempt + 1, 5))
            raise RuntimeError(f"Incomplete part {index}")
        count = (SIZE + chunk_size - 1) // chunk_size
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            for done, _ in enumerate(pool.map(download, range(count)), 1):
                if done % 20 == 0 or done == count:
                    print(f"BS-RoFormer {done}/{count}", flush=True)
        temporary = destination.with_suffix(".partial")
        with temporary.open("wb") as output:
            for index in range(count):
                with (parts / str(index)).open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        output.write(block)
        if temporary.stat().st_size != SIZE:
            raise RuntimeError("Asset length mismatch")
        temporary.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    config_name = NAME.replace(".ckpt", ".yaml")
    config_url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/" + config_name
    with urllib.request.urlopen(config_url, timeout=30) as response:
        config = response.read()
    (destination.parent / config_name).write_bytes(config)
    record = {"url": URL, "size": SIZE, "sha256": digest.hexdigest(),
              "hash_note": "Hash computed locally after download; upstream provides no published SHA256 here.",
              "config_url": config_url, "config_sha256": hashlib.sha256(config).hexdigest()}
    (ROOT / "configs/separation_model.lock.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
