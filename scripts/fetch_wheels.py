#!/usr/bin/env python3
"""Download official PyPI wheels with resumable byte ranges and SHA256 checking.

Used only for setup where the connection to files.pythonhosted.org is slow.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "cache/wheels"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch(spec):
    package, version = spec.split("==", 1)
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=30) as stream:
        data = json.load(stream)
    tags = {tag: rank for rank, tag in enumerate(sys_tags())}
    candidates = []
    for item in data["urls"]:
        if item["filename"].endswith(".whl"):
            matching = parse_wheel_filename(item["filename"])[3] & tags.keys()
            if matching:
                candidates.append((min(tags[tag] for tag in matching), item))
    if not candidates:
        raise RuntimeError(f"No compatible wheel for {spec}")
    item = min(candidates, key=lambda pair: pair[0])[1]
    # The official PyTorch CDN is faster than the PyPI CDN on this host.
    # This CUDA wheel's digest is published by download.pytorch.org/whl/cu128/torch/.
    if package == "torch" and version == "2.10.0":
        item = {"filename": "torch-2.10.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl",
                "url": "https://download-r2.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl",
                "size": 916750307,
                "digests": {"sha256": "1d01ffaebf64715c0f507a39463149cb19e596ff702bd4bcf862601f2881dabc"}}
    elif package == "triton" and version == "3.6.0":
        item["url"] = "https://download-r2.pytorch.org/whl/" + item["filename"]
        item["size"] = 188221377
        item["digests"] = {"sha256": "e021e87e8f266c6f87bf379b669c43c2800aac6247bdf5d518cb553e3c403c00"}
    elif package in {"nvidia-nccl-cu12", "nvidia-nvshmem-cu12"}:
        item["url"] = "https://pypi.nvidia.com/" + package + "/" + item["filename"]
        with urllib.request.urlopen(urllib.request.Request(item["url"], headers={"Range": "bytes=0-0"}), timeout=30) as response:
            item["size"] = int(response.headers["Content-Range"].split("/")[-1])
        item["digests"] = {"sha256": {
            "nvidia-nccl-cu12": "ad730cf15cb5d25fe849c6e6ca9eb5b76db16a80f13f425ac68d8e2e55624457",
            "nvidia-nvshmem-cu12": "042f2500f24c021db8a06c5eec2539027d57460e1c1a762055a6554f72c369bd",
        }[package]}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    target = DESTINATION / item["filename"]
    if target.exists() and digest(target) == item["digests"]["sha256"]:
        print(f"CACHED {target.name}", flush=True)
        return
    size, chunk_size = item["size"], 4 * 1024**2
    parts = DESTINATION / (target.name + ".parts")
    parts.mkdir(exist_ok=True)
    print(f"DOWNLOAD {target.name}: {size / 1024**2:.1f} MiB", flush=True)
    def part(index):
        start, end = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
        path = parts / str(index)
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        temporary = path.with_suffix(".tmp")
        for attempt in range(30):
            try:
                downloaded = temporary.stat().st_size if temporary.exists() else 0
                if downloaded == end - start + 1:
                    temporary.replace(path)
                    return
                position = start + downloaded
                request = urllib.request.Request(item["url"], headers={"Range": f"bytes={position}-{end}"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {position}-{end}/{size}":
                        raise RuntimeError(f"Unexpected range: HTTP {response.status}, {response.headers.get('Content-Range')}; expected {position}-{end}/{size}")
                    deadline = time.monotonic() + 15
                    with temporary.open("ab") as stream:
                        while block := response.read(64 * 1024):
                            stream.write(block)
                            if time.monotonic() >= deadline:
                                break
                if temporary.stat().st_size == end - start + 1:
                    temporary.replace(path)
                    return
            except Exception:
                if attempt == 29:
                    raise
                time.sleep(min(1 + attempt, 5))
        raise RuntimeError(f"Could not complete range {start}-{end}")
    count = (size + chunk_size - 1) // chunk_size
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for done, _ in enumerate(pool.map(part, range(count)), 1):
            if done % 20 == 0 or done == count:
                print(f"{package}: {done}/{count} ranges", flush=True)
    temporary = target.with_suffix(".partial")
    with temporary.open("wb") as out:
        for index in range(count):
            with (parts / str(index)).open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    out.write(block)
    if digest(temporary) != item["digests"]["sha256"]:
        raise RuntimeError(f"SHA256 mismatch for {target.name}")
    temporary.replace(target)
    print(f"VERIFIED {target.name}", flush=True)


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        fetch(spec)
