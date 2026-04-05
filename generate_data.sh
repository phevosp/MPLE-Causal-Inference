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

add_metadata() {
	local -n out_ref="$1"
	local key="$2"
	local value="$3"
	out_ref+=("--metadata" "${key}=${value}")
}

add_common_args() {
	local target_name="$1"
	local seed="$2"
	add_override "$target_name" global_params.N 1000
	add_override "$target_name" global_params.T 20
	add_override "$target_name" global_params.s 12
	add_override "$target_name" global_params.gamma_matrix_generator '"erdos_renyi"'
	add_override "$target_name" global_params.gamma_matrix_params.p 0.05
	add_override "$target_name" global_params.x_0_generator '"bernoulli"'
	add_override "$target_name" global_params.x_0_params.p 0.5
	add_override "$target_name" global_params.basis_params.num_shared_features 5
	add_override "$target_name" global_params.basis_params.shared_feature_seed "$seed"
	add_override "$target_name" generation_params.seed "$seed"
	add_override "$target_name" generation_params.gibbs_sweeps 10
	add_override "$target_name" estimation_params.beta 0.35
	add_override "$target_name" estimation_params.interaction_coefs '[0.35]'
	add_override "$target_name" estimation_params.eta 0.08
	add_override "$target_name" estimation_params.zeta -0.25
	add_override "$target_name" estimation_params.psi 0.20
	add_override "$target_name" global_params.basis_params.interaction_mode '"known_graph"'
}

add_uniform_tau_args() {
	local target_name="$1"
	local seed="$2"
	add_override "$target_name" global_params.basis_params.field_mode '"uniform"'
	add_override "$target_name" estimation_params.field_coefs '[0.30]'
	add_override "$target_name" estimation_params.tau_params.mode '"uniform_random"'
	add_override "$target_name" estimation_params.tau_params.lower -0.20
	add_override "$target_name" estimation_params.tau_params.upper 0.20
	add_override "$target_name" estimation_params.tau_params.seed "$seed"
}

add_shared_field_tau_args() {
	local target_name="$1"
	local seed="$2"
	add_override "$target_name" global_params.basis_params.field_mode '"shared_feature_field"'
	add_override "$target_name" estimation_params.field_coefs '[0.25,0.18,-0.10,0.12,-0.08,0.10,-0.06,0.08,-0.04,0.06,-0.02]'
	add_override "$target_name" estimation_params.tau_params.mode '"uniform_random"'
	add_override "$target_name" estimation_params.tau_params.lower -0.25
	add_override "$target_name" estimation_params.tau_params.upper 0.25
	add_override "$target_name" estimation_params.tau_params.seed "$seed"
}

declare -a EXPERIMENT_LABELS=(
	"tau_uniform_field"
	"tau_shared_field"
	"tau_shared_field_alt_seed"
)

seed_counter=600

for label in "${EXPERIMENT_LABELS[@]}"; do
	args=()
	add_metadata args suite tau_short_test
	add_metadata args test_type "$label"
	add_common_args args "$seed_counter"

	case "$label" in
		tau_uniform_field)
			add_metadata args field_complexity uniform
			add_metadata args tau_mode uniform_random
			add_uniform_tau_args args "$seed_counter"
			;;
		tau_shared_field)
			add_metadata args field_complexity shared_feature_field
			add_metadata args tau_mode uniform_random
			add_shared_field_tau_args args "$seed_counter"
			;;
		tau_shared_field_alt_seed)
			add_metadata args field_complexity shared_feature_field
			add_metadata args tau_mode uniform_random
			add_shared_field_tau_args args "$((seed_counter + 1000))"
			add_override args global_params.gamma_matrix_params.p 0.08
			add_override args estimation_params.interaction_coefs '[0.45]'
			;;
		*)
			echo "Unknown experiment label: $label" >&2
			exit 1
			;;
	esac

	run_generation "$label" "${args[@]}"
	seed_counter=$((seed_counter + 1))
done

echo "Finished generating tau smoke-test datasets."
echo "Manifest: $MANIFEST_PATH"
