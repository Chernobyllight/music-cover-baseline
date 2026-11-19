"""Apply the macOS patches to an isolated copy of the pinned upstream code."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATCH_DIRECTORY = ROOT / "patches/macos_mps"
REVISIONS = {
    "YuE": "1647252d68b70cbead046ffd3f7caf036dde47ee",
    "seed-vc": "51383efd921027683c89e5348211d93ff12ac2a8",
}
PATCHES = (
    ("seed-vc", "seed-vc-mps.patch"),
    ("YuE", "yue2-mps-memory.patch"),
    ("YuE", "yue2-mps-sampling.patch"),
    ("YuE", "yue2-native-ar-gqa.patch"),
    ("YuE", "yue2-mps-semantic-head.patch"),
    ("YuE", "yue2-mps-nar-query-blocks.patch"),
    ("seed-vc", "seed-vc-mps-dit-fp16.patch"),
    ("seed-vc", "seed-vc-key-mask.patch"),
    ("seed-vc", "seed-vc-equal-heads.patch"),
    ("seed-vc", "seed-vc-soundfile.patch"),
    ("seed-vc", "seed-vc-mps-bigvgan-conv.patch"),
)


def apply(checkout):
    for name, patch in PATCHES:
        revision = REVISIONS[name]
        directory = checkout / "third_party" / name
        command = ["git", "-C", str(directory)]
        actual = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Refusing {patch}: expected {revision}, got {actual}")
        path = str(PATCH_DIRECTORY / patch)
        if subprocess.run([*command, "apply", "--reverse", "--check", path], capture_output=True).returncode == 0:
            print(f"Already applied: {patch}")
            continue
        subprocess.run([*command, "apply", "--check", path], check=True)
        subprocess.run([*command, "apply", path], check=True)
        print(f"Applied: {patch}")
