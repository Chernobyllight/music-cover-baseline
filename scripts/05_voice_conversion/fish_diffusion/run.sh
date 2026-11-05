#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
exec /opt/conda/bin/python -m music_cover.pipeline --stage 05_voice_conversion "$@" --voice-backend fish_diffusion --voice-mode finetuned
