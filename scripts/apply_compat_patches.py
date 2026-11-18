#!/usr/bin/env python3
"""Apply reviewed patches only to the pinned upstream commits; safe to rerun."""
from pathlib import Path
import argparse
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATCHES = [("seed-vc", "51383efd921027683c89e5348211d93ff12ac2a8", "seed-vc-mps.patch"),
           ("YuE", "1647252d68b70cbead046ffd3f7caf036dde47ee", "yue2-mps-memory.patch"),
           ("YuE", "1647252d68b70cbead046ffd3f7caf036dde47ee", "yue2-mps-sampling.patch")]
CANDIDATES = {
    "seed-vc-mps-dit-fp16": ("seed-vc", "51383efd921027683c89e5348211d93ff12ac2a8", "candidates/seed-vc-mps-dit-fp16.patch"),
    "yue2-mps-semantic-head": ("YuE", "1647252d68b70cbead046ffd3f7caf036dde47ee", "candidates/yue2-mps-semantic-head.patch"),
    "seed-vc-key-mask": ("seed-vc", "51383efd921027683c89e5348211d93ff12ac2a8", "candidates/seed-vc-key-mask.patch"),
    "seed-vc-equal-heads": ("seed-vc", "51383efd921027683c89e5348211d93ff12ac2a8", "candidates/seed-vc-equal-heads.patch"),
    "seed-vc-loop": ("seed-vc", "51383efd921027683c89e5348211d93ff12ac2a8", "candidates/seed-vc-loop.patch"),
    "yue2-native-ar-gqa": ("YuE", "1647252d68b70cbead046ffd3f7caf036dde47ee", "candidates/yue2-native-ar-gqa.patch"),
    "seed-vc-soundfile": ("seed-vc", "51383efd921027683c89e5348211d93ff12ac2a8", "candidates/seed-vc-soundfile.patch"),
}


# The combination measured together in docs/15-mps-full-song.md.
OPTIMIZED = (
    "yue2-native-ar-gqa", "yue2-mps-semantic-head", "seed-vc-mps-dit-fp16",
    "seed-vc-key-mask", "seed-vc-equal-heads", "seed-vc-soundfile",
)


def apply(candidates=(), *, checkout=None):
    for name, revision, patch in [*PATCHES, *(CANDIDATES[name] for name in candidates)]:
        directory = (checkout or ROOT) / "third_party" / name
        command = ["git", "-C", str(directory)]
        actual = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Refusing {patch}: expected {revision}, got {actual}")
        path = str(ROOT / "patches" / patch)
        if subprocess.run([*command, "apply", "--reverse", "--check", path], capture_output=True).returncode == 0:
            print(f"Already applied: {patch}")
            continue
        subprocess.run([*command, "apply", "--check", path], check=True)
        subprocess.run([*command, "apply", path], check=True)
        print(f"Applied: {patch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", choices=sorted(CANDIDATES), default=[],
                        help="Explicitly apply an experimental patch; omitted by normal installation")
    apply(parser.parse_args().candidate)
