#!/usr/bin/env python3
"""Assemble the validated MPS checkout from installed environments and pinned sources."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

from apply_compat_patches import CANDIDATES, OPTIMIZED, PATCHES, ROOT, apply


def prepare():
    environments = {name: ROOT / "envs" / name for name in ("control", "sheetsage2")}
    environments.update({name: ROOT / "envs" / f"{name}-torch215"
                         for name in ("yue2", "separator", "seed-vc")})
    missing = [str(path) for path in environments.values() if not (path / "bin/python").is_file()]
    if missing:
        raise SystemExit("Install the optimized environments via scripts/setup_macos.sh --optimized:\n"
                         + "\n".join(missing))

    patches = [*PATCHES, *(CANDIDATES[name] for name in OPTIMIZED)]
    revisions = {name: revision for name, revision, _ in patches}
    scripts = [ROOT / "scripts" / name for name in ("run_baseline.sh", "stage_runner.py")]
    sources = scripts + sorted((ROOT / "src").rglob("*.py"))
    separator_patch = ROOT / "patches/candidates/separator-mps-axial-fp16.patch"
    digest = hashlib.sha256(separator_patch.read_bytes())
    for name, revision, patch in patches:
        digest.update(f"{name}/{revision}/{patch}\n".encode())
        digest.update((ROOT / "patches" / patch).read_bytes())
    for path in [*sources, Path(__file__), Path(__file__).with_name("apply_compat_patches.py")]:
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
            # Patch only the staged separator adapter. The base environment shares src/.
            subprocess.run(["patch", "--batch", "-F", "0", "-p1", "-i", str(separator_patch)],
                           cwd=staging, check=True)
            for name, revision in revisions.items():
                target = staging / "third_party" / name
                subprocess.run(["git", "clone", "--quiet", "--local", "--no-checkout",
                                str(ROOT / "third_party" / name), str(target)], check=True)
                subprocess.run(["git", "-C", str(target), "checkout", "--quiet", "--detach", revision],
                               check=True)
            apply(OPTIMIZED, checkout=staging)
            try:
                staging.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
        link = Path(tmp) / "optimized"
        link.symlink_to(destination.name)
        link.replace(destination.parent / "mps-optimized")
    print("Optimized workspace ready: envs/mps-optimized")


if __name__ == "__main__":
    prepare()
