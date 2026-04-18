#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GENERATION_MANIFEST_PATH="${GENERATION_MANIFEST_PATH:-$SCRIPT_DIR/experiments/SyntheticHybridExperiments/generation_manifest.csv}"
FIT_MANIFEST_PATH="${FIT_MANIFEST_PATH:-$SCRIPT_DIR/experiments/SyntheticHybridExperiments/fit_manifest.csv}"
TARGET_PAIRS_PATH="${TARGET_PAIRS_PATH:-$SCRIPT_DIR/data/configs/posterior_predictive_target_pairs.csv}"
POSTERIOR_PREDICTIVE_SPEC_PATH="${POSTERIOR_PREDICTIVE_SPEC_PATH:-$SCRIPT_DIR/data/configs/posterior_predictive_spec.yaml}"
OVERWRITE_FLAG="${POSTERIOR_PREDICTIVE_OVERWRITE:-0}"

if command -v pixi >/dev/null 2>&1; then
	RUNNER=(pixi run python -u)
else
	RUNNER=(python -u)
fi

ARGS=(
	--generation_manifest_path "$GENERATION_MANIFEST_PATH"
	--fit_manifest_path "$FIT_MANIFEST_PATH"
	--target_pairs_path "$TARGET_PAIRS_PATH"
	--spec_path "$POSTERIOR_PREDICTIVE_SPEC_PATH"
)
if [[ "$OVERWRITE_FLAG" == "1" ]]; then
	ARGS+=(--overwrite)
fi

"${RUNNER[@]}" run_posterior_predictive_pipeline.py "${ARGS[@]}"
