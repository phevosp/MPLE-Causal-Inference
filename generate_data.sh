#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MANIFEST_PATH="${EXPERIMENT_MANIFEST:-$SCRIPT_DIR/experiments/latest_manifest.txt}"
BASE_CONFIG="base_config.yaml"

mkdir -p "$(dirname "$MANIFEST_PATH")"
: >"$MANIFEST_PATH"

if command -v pixi >/dev/null 2>&1; then
	RUNNER=(pixi run python -u)
else
	RUNNER=(python -u)
fi

run_generation() {
	local label="$1"
	shift
	"${RUNNER[@]}" data/synthetic_data_generation.py \
		--config_name "$BASE_CONFIG" \
		--descriptor "$label" \
		--manifest_path "$MANIFEST_PATH" \
		"$@"
	sleep 1
}

COMMON_ARGS=(
	--config_override global_params.N=300
	--config_override global_params.T=20
	--config_override global_params.s=12
	--config_override global_params.gamma_matrix_generator='"erdos_renyi"'
	--config_override global_params.gamma_matrix_params.p=0.05
	--config_override global_params.x_0_generator='"bernoulli"'
	--config_override global_params.x_0_params.p=0.5
	--config_override generation_params.gibbs_sweeps=10
	--config_override generation_params.intervention_mode='"generated_z"'
	--config_override estimation_params.beta=0.35
	--config_override estimation_params.xi=0.25
	--config_override estimation_params.eta=0.08
	--config_override estimation_params.zeta=-0.25
	--config_override estimation_params.psi=0.20
)

run_generation "generated_z_uniform_field" \
	"${COMMON_ARGS[@]}" \
	--config_override generation_params.seed=600 \
	--config_override global_params.basis_params.field_mode='"uniform"' \
	--config_override estimation_params.field_coefs='[]' \
	--config_override estimation_params.tau_params.mode='"uniform_random"' \
	--config_override estimation_params.tau_params.lower=-0.20 \
	--config_override estimation_params.tau_params.upper=0.20 \
	--config_override estimation_params.tau_params.seed=600

run_generation "generated_z_shared_feature_field" \
	"${COMMON_ARGS[@]}" \
	--config_override generation_params.seed=601 \
	--config_override global_params.basis_params.field_mode='"shared_feature_field"' \
	--config_override global_params.basis_params.num_shared_features=3 \
	--config_override global_params.basis_params.shared_feature_seed=601 \
	--config_override estimation_params.field_coefs='[0.20,-0.10,0.15,-0.08,0.12,-0.05]' \
	--config_override estimation_params.tau_params.mode='"uniform_random"' \
	--config_override estimation_params.tau_params.lower=-0.20 \
	--config_override estimation_params.tau_params.upper=0.20 \
	--config_override estimation_params.tau_params.seed=601

run_generation "generated_z_latent_field" \
	"${COMMON_ARGS[@]}" \
	--config_override generation_params.seed=602 \
	--config_override global_params.basis_params.field_mode='"latent_feature_matrix"' \
	--config_override global_params.basis_params.latent_rank=4

if [[ -n "${FIXED_Z_PANEL_PATH:-}" && -n "${FIXED_Z_Z0_PATH:-}" ]]; then
	run_generation "fixed_z_uniform_field" \
		"${COMMON_ARGS[@]}" \
		--config_override generation_params.seed=603 \
		--config_override global_params.basis_params.field_mode='"uniform"' \
		--config_override estimation_params.field_coefs='[]' \
		--config_override estimation_params.tau_params.mode='"uniform_random"' \
		--config_override estimation_params.tau_params.lower=-0.20 \
		--config_override estimation_params.tau_params.upper=0.20 \
		--config_override estimation_params.tau_params.seed=603 \
		--config_override generation_params.intervention_mode='"fixed_z"' \
		--config_override generation_params.fixed_z_source.panel_path="\"${FIXED_Z_PANEL_PATH}\"" \
		--config_override generation_params.fixed_z_source.z0_path="\"${FIXED_Z_Z0_PATH}\""
fi

echo "Finished generating synthetic datasets."
echo "Manifest: $MANIFEST_PATH"
