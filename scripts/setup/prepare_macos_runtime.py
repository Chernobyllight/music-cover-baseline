#!/usr/bin/env python3
"""Assemble the validated MPS checkout from installed environments and pinned sources."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys

from apply_macos_patches import PATCH_DIRECTORY, PATCHES, REVISIONS, ROOT, apply

sys.path.insert(0, str(ROOT / "src"))
from music_cover.common.macos import ENVIRONMENTS, WORKSPACE


def prepare():
    environments = {name: ROOT / "envs" / target for name, target in ENVIRONMENTS.items()}
    missing = [str(path) for path in environments.values() if not (path / "bin/python").is_file()]
    if missing:
        raise SystemExit("Install the macOS environments via scripts/setup/macos.sh:\n"
                         + "\n".join(missing))

    scripts = [ROOT / "scripts" / name for name in ("run_baseline.sh", "stage_runner.py")]
    sources = scripts + sorted((ROOT / "src").rglob("*.py"))
    separator_patch = PATCH_DIRECTORY / "separator-mps-axial-fp16.patch"
    digest = hashlib.sha256(separator_patch.read_bytes())
    for name, patch in PATCHES:
        digest.update(f"{name}/{REVISIONS[name]}/{patch}\n".encode())
        digest.update((PATCH_DIRECTORY / patch).read_bytes())
    for path in [*sources, Path(__file__), Path(__file__).with_name("apply_macos_patches.py")]:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    destination = ROOT / "envs" / ("mps-optimized-" + digest.hexdigest()[:16])
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".mps-optimized-") as tmp:
        staging = Path(tmp) / "checkout"
        if not destination.exists():
            (staging / "envs").mkdir(parents=True)
            for name, path in environments.items():
                (staging / "envs" / name).symlink_to(path)
            for name in ("configs", "data", "cache", "models", "runs"):
                (staging / name).symlink_to(ROOT / name)
            for path in sources:
                target = staging / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            # Patch the staged separator adapter without changing the shared source tree.
            subprocess.run(["patch", "--batch", "-F", "0", "-p1", "-i", str(separator_patch)],
                           cwd=staging, check=True)
            for name, revision in REVISIONS.items():
                target = staging / "third_party" / name
                subprocess.run(["git", "clone", "--quiet", "--local", "--no-checkout",
                                str(ROOT / "third_party" / name), str(target)], check=True)
                subprocess.run(["git", "-C", str(target), "checkout", "--quiet", "--detach", revision],
                               check=True)
            apply(staging)
            try:
                staging.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
        link = Path(tmp) / "optimized"
        link.symlink_to(destination.name)
        link.replace(ROOT / WORKSPACE)
    print(f"macOS MPS workspace ready: {WORKSPACE}")


if __name__ == "__main__":
    prepare()
