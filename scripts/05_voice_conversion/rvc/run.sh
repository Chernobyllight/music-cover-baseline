#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${PROJECT_ROOT}/scripts/run_baseline.sh" --stage 05_voice_conversion "$@" --voice-backend rvc --voice-mode finetuned
