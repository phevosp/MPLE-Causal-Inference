#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_CONFIG="base_config.yaml"
MANIFEST_PATH="${EXPERIMENT_MANIFEST:-$SCRIPT_DIR/experiments/latest_manifest.txt}"

mkdir -p "$(dirname "$MANIFEST_PATH")"
: >"$MANIFEST_PATH"

if command -v pixi >/dev/null 2>&1; then
	RUNNER=(pixi run python -u)
elif command -v python >/dev/null 2>&1; then
	RUNNER=(python -u)
else
	echo "Error: neither 'pixi' nor 'python' is available in PATH." >&2
	exit 1
fi

run_generation() {
	local label="$1"
	shift
	echo "Generating ${label}..."
	"${RUNNER[@]}" data/synthetic_data_generation.py \
		--config_name "$BASE_CONFIG" \
		--descriptor "$label" \
		--manifest_path "$MANIFEST_PATH" \
		"$@"
	sleep 1
}

add_override() {
	local -n out_ref="$1"
	local key="$2"
	local value="$3"
	out_ref+=("--config_override" "${key}=${value}")
}

common_generated_z_args() {
	local target_name="$1"
	local seed="$2"
	add_override "$target_name" global_params.N 300
	add_override "$target_name" global_params.T 20
	add_override "$target_name" global_params.s 12
	add_override "$target_name" global_params.gamma_matrix_generator '"erdos_renyi"'
	add_override "$target_name" global_params.gamma_matrix_params.p 0.05
	add_override "$target_name" global_params.x_0_generator '"bernoulli"'
	add_override "$target_name" global_params.x_0_params.p 0.5
	add_override "$target_name" global_params.basis_params.interaction_mode '"known_graph"'
	add_override "$target_name" generation_params.seed "$seed"
	add_override "$target_name" generation_params.gibbs_sweeps 10
	add_override "$target_name" generation_params.intervention_mode '"generated_z"'
	add_override "$target_name" estimation_params.beta 0.35
	add_override "$target_name" estimation_params.interaction_coefs '[0.25]'
	add_override "$target_name" estimation_params.eta 0.08
	add_override "$target_name" estimation_params.zeta -0.25
	add_override "$target_name" estimation_params.psi 0.20
}

uniform_args=()
common_generated_z_args uniform_args 600
add_override uniform_args global_params.basis_params.field_mode '"uniform"'
add_override uniform_args estimation_params.field_coefs '[]'
add_override uniform_args estimation_params.tau_params.mode '"uniform_random"'
add_override uniform_args estimation_params.tau_params.lower -0.20
add_override uniform_args estimation_params.tau_params.upper 0.20
add_override uniform_args estimation_params.tau_params.seed 600
run_generation "generated_z_uniform_field" "${uniform_args[@]}"

shared_args=()
common_generated_z_args shared_args 601
add_override shared_args global_params.basis_params.field_mode '"shared_feature_field"'
add_override shared_args global_params.basis_params.num_shared_features 3
add_override shared_args global_params.basis_params.shared_feature_seed 601
add_override shared_args estimation_params.field_coefs '[0.20,-0.10,0.15,-0.08,0.12,-0.05]'
add_override shared_args estimation_params.tau_params.mode '"uniform_random"'
add_override shared_args estimation_params.tau_params.lower -0.20
add_override shared_args estimation_params.tau_params.upper 0.20
add_override shared_args estimation_params.tau_params.seed 601
run_generation "generated_z_shared_feature_field" "${shared_args[@]}"

if [[ -n "${FIXED_Z_PANEL_PATH:-}" && -n "${FIXED_Z_Z0_PATH:-}" ]]; then
	fixed_args=()
	common_generated_z_args fixed_args 900
	add_override fixed_args global_params.basis_params.field_mode '"uniform"'
	add_override fixed_args estimation_params.field_coefs '[]'
	add_override fixed_args estimation_params.tau_params.mode '"fixed"'
	add_override fixed_args estimation_params.tau_params.vector '[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]'
	add_override fixed_args generation_params.intervention_mode '"fixed_z"'
	add_override fixed_args generation_params.fixed_z_source.panel_path "\"${FIXED_Z_PANEL_PATH}\""
	add_override fixed_args generation_params.fixed_z_source.z0_path "\"${FIXED_Z_Z0_PATH}\""
	if [[ -n "${FIXED_Z_SHARED_PANEL_DIR:-}" ]]; then
		add_override fixed_args generation_params.fixed_z_source.shared_panel_dir "\"${FIXED_Z_SHARED_PANEL_DIR}\""
	fi
	run_generation "fixed_z_uniform_field" "${fixed_args[@]}"
fi

echo "Finished generating synthetic datasets."
echo "Manifest: $MANIFEST_PATH"
