#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SPEC_PATH="${GENERATION_SPEC_PATH:-$SCRIPT_DIR/data/configs/generation_spec.yaml}"
OVERWRITE_FLAG="${GENERATION_OVERWRITE:-0}"

if command -v pixi >/dev/null 2>&1; then
	RUNNER=(pixi run python -u)
else
	RUNNER=(python -u)
fi

ARGS=(--spec_path "$SPEC_PATH")
if [[ "$OVERWRITE_FLAG" == "1" ]]; then
	ARGS+=(--overwrite)
fi

"${RUNNER[@]}" run_generation_pipeline.py "${ARGS[@]}"
