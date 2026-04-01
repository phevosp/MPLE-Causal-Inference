#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MANIFEST_PATH="${EXPERIMENT_MANIFEST:-$SCRIPT_DIR/experiments/latest_manifest.txt}"
REPORT_STEM="${EXPERIMENT_REPORT_STEM:-$SCRIPT_DIR/reports/conditional_experiment_report}"

if command -v pixi >/dev/null 2>&1; then
	RUNNER=(pixi run python -u)
elif command -v python >/dev/null 2>&1; then
	RUNNER=(python -u)
else
	echo "Error: neither 'pixi' nor 'python' is available in PATH." >&2
	exit 1
fi

mkdir -p "$(dirname "$REPORT_STEM")"

EXPERIMENT_MANIFEST="$MANIFEST_PATH" "$SCRIPT_DIR/generate_data.sh"

if [ ! -f "$MANIFEST_PATH" ]; then
	echo "Manifest not found: $MANIFEST_PATH" >&2
	exit 1
fi

while IFS= read -r data_folder; do
	if [ -z "$data_folder" ]; then
		continue
	fi

	if [ ! -d "$data_folder" ]; then
		echo "Skipping missing experiment folder: $data_folder"
		continue
	fi

	echo "Running conditional MPLE for ${data_folder}..."
	"${RUNNER[@]}" mple.py --data_folder "$data_folder" "$@"
done <"$MANIFEST_PATH"

echo "Building report..."
"${RUNNER[@]}" report_experiments.py --manifest "$MANIFEST_PATH" --report_stem "$REPORT_STEM"

echo "Finished running MPLE across manifest experiments."
echo "Report: ${REPORT_STEM}.md"
