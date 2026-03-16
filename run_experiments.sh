#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIGS=(
	"conditional_config.yaml"
	"conditional_config_p_large.yaml"
	"ising_config.yaml"
	"ising_config_p_large.yaml"
)

if command -v pixi >/dev/null 2>&1; then
	RUNNER=(pixi run python -u)
elif command -v python >/dev/null 2>&1; then
	RUNNER=(python -u)
else
	echo "Error: neither 'pixi' nor 'python' is available in PATH." >&2
	exit 1
fi

for config in "${CONFIGS[@]}"; do
	echo "Generating data for ${config}..."
	"${RUNNER[@]}" data/synthetic_data_generation.py --config_name "$config"
done

echo "Finished generating all datasets."

shopt -s nullglob

EXPERIMENT_DIRS=("$SCRIPT_DIR"/experiments/*)

if [ ${#EXPERIMENT_DIRS[@]} -eq 0 ]; then
	echo "No experiment folders found in experiments/."
	exit 0
fi

for data_folder in "${EXPERIMENT_DIRS[@]}"; do
	if [ ! -d "$data_folder" ]; then
		continue
	fi

	if [ ! -f "$data_folder/realized_config.yaml" ] || [ ! -f "$data_folder/synthetic_data.npz" ] || [ ! -f "$data_folder/gamma_matrix.npy" ] || [ ! -f "$data_folder/x_0.npy" ]; then
		echo "Skipping ${data_folder}: missing required experiment files."
		continue
	fi

	echo "Running MPLE for ${data_folder}..."
	"${RUNNER[@]}" mple.py --data_folder "$data_folder" "$@"
done

echo "Finished running MPLE across experiment folders."
