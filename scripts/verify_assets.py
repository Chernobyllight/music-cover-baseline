#!/usr/bin/env python3
"""Check restored weights/code against the original baseline's SHA256 records."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    expected = json.loads((ROOT / "configs/macos/local_assets.lock.json").read_text())
    failures = []
    checked = 0
    for name, record in expected.items():
        if not name.startswith("models/"):
            continue
        path = ROOT / name
        if not path.is_file():
            failures.append(f"Missing: {name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            failures.append(f"SHA256 mismatch: {name}")
        checked += 1
    if failures:
        raise RuntimeError("Asset verification failed:\n" + "\n".join(failures))
    print(f"Verified {checked} model assets against the baseline SHA256 manifest")


if __name__ == "__main__":
    main()
