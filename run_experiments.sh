#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MANIFEST_PATH="${EXPERIMENT_MANIFEST:-$SCRIPT_DIR/experiments/SyntheticExperimentsGrid/latest_manifest.txt}"
COMPLETED_MANIFEST_PATH="${EXPERIMENT_COMPLETED_MANIFEST:-$SCRIPT_DIR/experiments/SyntheticExperimentsGrid/completed_manifest.txt}"
REPORT_STEM="${EXPERIMENT_REPORT_STEM:-$SCRIPT_DIR/experiments/SyntheticExperimentsGrid/reports/conditional_experiment_report}"
SKIP_GENERATION=0
MPLE_ARGS=()

while [ "$#" -gt 0 ]; do
	case "$1" in
		--skip-generation)
			SKIP_GENERATION=1
			shift
			;;
		--help|-h)
			echo "Usage: bash run_experiments.sh [--skip-generation] [mple.py args...]"
			echo
			echo "  --skip-generation  Reuse the existing manifest instead of rerunning generate_data.sh"
			echo "  Remaining arguments are passed through to mple.py for every experiment."
			exit 0
			;;
		*)
			MPLE_ARGS+=("$1")
			shift
			;;
	esac
done

if command -v pixi >/dev/null 2>&1; then
	RUNNER=(pixi run python -u)
elif command -v python >/dev/null 2>&1; then
	RUNNER=(python -u)
else
	echo "Error: neither 'pixi' nor 'python' is available in PATH." >&2
	exit 1
fi

mkdir -p "$(dirname "$REPORT_STEM")"
mkdir -p "$(dirname "$COMPLETED_MANIFEST_PATH")"

if [ "$SKIP_GENERATION" -eq 0 ]; then
	EXPERIMENT_MANIFEST="$MANIFEST_PATH" "$SCRIPT_DIR/generate_data.sh"
else
	echo "Skipping data generation and reusing manifest: $MANIFEST_PATH"
fi

if [ ! -f "$MANIFEST_PATH" ]; then
	echo "Manifest not found: $MANIFEST_PATH" >&2
	exit 1
fi

: >"$COMPLETED_MANIFEST_PATH"

while IFS= read -r data_folder; do
	data_folder="${data_folder%$'\r'}"

	if [ -z "$data_folder" ]; then
		continue
	fi

	if [ ! -d "$data_folder" ]; then
		echo "Skipping missing experiment folder: $data_folder"
		continue
	fi

	echo "Running conditional MPLE for ${data_folder}..."
	if "${RUNNER[@]}" mple.py --data_folder "$data_folder" "${MPLE_ARGS[@]}"; then
		if [ -f "$data_folder/mple_summary.csv" ] && [ -f "$data_folder/mple_summary.md" ]; then
			printf '%s\n' "$data_folder" >>"$COMPLETED_MANIFEST_PATH"
		else
			echo "Skipping report manifest append; summary outputs missing for $data_folder" >&2
		fi
	else
		echo "MPLE failed for $data_folder; not adding to completed manifest." >&2
	fi
done <"$MANIFEST_PATH"

if [ ! -s "$COMPLETED_MANIFEST_PATH" ]; then
	echo "No completed experiments were available for reporting." >&2
	exit 1
fi

echo "Building report..."
"${RUNNER[@]}" report_parameter_recovery_detailed.py --manifest "$COMPLETED_MANIFEST_PATH" --report_stem "$REPORT_STEM"

echo "Finished running MPLE across manifest experiments."
echo "Report: ${REPORT_STEM}.md"
echo "Completed manifest: ${COMPLETED_MANIFEST_PATH}"
