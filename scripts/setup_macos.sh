#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -gt 0 && ( $# -ne 1 || "$1" != --optimized ) ]]; then
    echo "Usage: bash scripts/setup_macos.sh" >&2
    exit 2
fi
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Requires native Apple Silicon macOS' >&2; exit 1; }
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/cache/python}"
export PYTORCH_ENABLE_MPS_FALLBACK=0
export HF_HOME="$ROOT/cache/huggingface"
command -v ffprobe >/dev/null && ffprobe -version >/dev/null || { echo "Install FFmpeg with ffprobe (brew install ffmpeg)" >&2; exit 1; }
git submodule update --init -- third_party/YuE third_party/seed-vc
for target in control sheetsage2 yue2-torch215 separator-torch215 seed-vc-torch215; do
    lock="configs/macos/$target.txt"
    if [[ "$target" == *-torch215 ]]; then
        lock="configs/macos/candidates/$target.txt"
    fi
    if [[ ! -x "envs/$target/bin/python" ]]; then
        uv venv --python 3.11 "envs/$target"
    fi
    uv pip sync --require-hashes --python "envs/$target/bin/python" "$lock"
done
uv pip install --python envs/yue2-torch215/bin/python --no-deps -e third_party/YuE
envs/control/bin/python scripts/prepare_macos_optimized.py
workspace="$ROOT/envs/mps-optimized"
envs/control/bin/python - <<'PY'
from pathlib import Path
import imageio_ffmpeg
path = Path('envs/control/bin/ffmpeg')
if not path.exists():
    path.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
PY
export PATH="$ROOT/envs/control/bin:$PATH"
export PYTHONPATH="$workspace/third_party/YuE/src${PYTHONPATH:+:$PYTHONPATH}"
for name in control sheetsage2 yue2 separator seed-vc; do
    "$workspace/envs/$name/bin/python" "$workspace/scripts/stage_runner.py" probe --device mps --environment "$name"
done
echo 'Environments ready. Restore weights with scripts/download_{models,voice_models,separator_model}.py and scripts/prepare_demo.py.'
