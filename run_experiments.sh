#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MANIFEST_PATH="${GENERATION_MANIFEST_PATH:-$SCRIPT_DIR/experiments/SyntheticHybridExperiments/generation_manifest.csv}"
FITS_SPEC_PATH="${FITS_SPEC_PATH:-$SCRIPT_DIR/data/configs/fits_spec.yaml}"
OVERWRITE_FLAG="${FIT_OVERWRITE:-0}"

if command -v pixi >/dev/null 2>&1; then
	RUNNER=(pixi run python -u)
else
	RUNNER=(python -u)
fi

ARGS=(
	--manifest_path "$MANIFEST_PATH"
	--fits_spec_path "$FITS_SPEC_PATH"
)
if [[ "$OVERWRITE_FLAG" == "1" ]]; then
	ARGS+=(--overwrite)
fi

"${RUNNER[@]}" run_fit_pipeline.py "${ARGS[@]}"
