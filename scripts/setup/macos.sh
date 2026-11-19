#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/setup/macos.sh" >&2
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
install_environment() {
    local target="$1"
    if [[ ! -x "envs/$target/bin/python" ]]; then
        uv venv --python 3.11 "envs/$target"
    fi
    uv pip sync --require-hashes --python "envs/$target/bin/python" "$2"
}
install_environment control configs/macos/control.txt
while read -r name target lock; do
    if [[ "$name" != control ]]; then
        install_environment "$target" "$lock"
    fi
    if [[ "$name" == yue2 ]]; then
        yue_target="$target"
    fi
done < <(PYTHONPATH="$ROOT/src" envs/control/bin/python - <<'PYTHON'
from pathlib import Path
from music_cover.common.macos import ENVIRONMENTS, lock_path

for name, target in ENVIRONMENTS.items():
    print(name, target, lock_path(Path("."), name))
PYTHON
)
uv pip install --python "envs/$yue_target/bin/python" --no-deps -e third_party/YuE
envs/control/bin/python scripts/setup/prepare_macos_runtime.py
workspace="$ROOT/$(PYTHONPATH="$ROOT/src" envs/control/bin/python -c 'from music_cover.common.macos import WORKSPACE; print(WORKSPACE)')"
envs/control/bin/python - <<'PY'
from pathlib import Path
import imageio_ffmpeg
path = Path('envs/control/bin/ffmpeg')
if not path.exists():
    path.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
PY
export PATH="$ROOT/envs/control/bin:$PATH"
export PYTHONPATH="$workspace/third_party/YuE/src${PYTHONPATH:+:$PYTHONPATH}"
for env_dir in "$workspace"/envs/*; do
    name="${env_dir##*/}"
    "$env_dir/bin/python" "$workspace/scripts/stage_runner.py" probe --device mps --environment "$name"
done
echo 'Environments ready. Restore weights with scripts/download_{models,voice_models,separator_model}.py and scripts/prepare_demo.py.'
