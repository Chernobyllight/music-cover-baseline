#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
case "${1:-}" in
    "") ;;
    --optimized) ;;
    *) echo "Usage: bash scripts/setup_macos.sh [--optimized]" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "Expected at most one option" >&2; exit 2; }
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Requires native Apple Silicon macOS' >&2; exit 1; }
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/cache/python}"
export PYTORCH_ENABLE_MPS_FALLBACK=0
export HF_HOME="$ROOT/cache/huggingface"
git submodule update --init --recursive
for name in control sheetsage2 yue2 separator seed-vc; do
    target="$name"
    lock="configs/macos/$name.txt"
    if [[ "${1:-}" == --optimized && "$name" != control && "$name" != sheetsage2 ]]; then
        target="$name-torch215"
        lock="configs/macos/candidates/$target.txt"
    fi
    if [[ ! -x "envs/$target/bin/python" ]]; then
        uv venv --python 3.11 "envs/$target"
    fi
    uv pip sync --require-hashes --python "envs/$target/bin/python" "$lock"
done
if [[ "${1:-}" == --optimized ]]; then
    uv pip install --python envs/yue2-torch215/bin/python --no-deps -e third_party/YuE
    envs/control/bin/python scripts/prepare_macos_optimized.py
    workspace="$ROOT/envs/mps-optimized"
else
    uv pip install --python envs/yue2/bin/python --no-deps -e third_party/YuE
    envs/control/bin/python scripts/apply_compat_patches.py
    workspace="$ROOT"
fi
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
