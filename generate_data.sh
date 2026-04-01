#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_CONFIG="base_config.yaml"

run_generation() {
	local label="$1"
	shift

	echo "Generating ${label}..."
	pixi run python -u data/synthetic_data_generation.py --config_name "$BASE_CONFIG" "$@"
	sleep 1
}

COMMON_GENERATION=(
	--config_override generation_params.gibbs_sweeps=25
)

run_generation "baseline" "${COMMON_GENERATION[@]}"

run_generation "p_large" "${COMMON_GENERATION[@]}" \
	--config_override global_params.gamma_matrix_params.p=0.1

run_generation "N_large" "${COMMON_GENERATION[@]}" \
	--config_override global_params.N=5000

run_generation "params_large" "${COMMON_GENERATION[@]}" \
	--config_override 'estimation_params.field_coefs=[1,-0.3,0.2]' \
	--config_override estimation_params.beta=1 \
	--config_override 'estimation_params.interaction_coefs=[0.5,0.2,-0.1]' \
	--config_override estimation_params.eta=0.1 \
	--config_override estimation_params.zeta=-0.5 \
	--config_override estimation_params.psi=0.5

run_generation "robustness" "${COMMON_GENERATION[@]}" \
	--config_override global_params.N=5000 \
	--config_override global_params.T=10 \
	--config_override global_params.s=0 \
	--config_override global_params.gamma_matrix_generator=complete \
	--config_override global_params.x_0_params.p=0.2 \
	--config_override 'estimation_params.field_coefs=[1,-0.5,0.25]' \
	--config_override estimation_params.beta=-1 \
	--config_override 'estimation_params.interaction_coefs=[1,0.3,-0.2]' \
	--config_override estimation_params.eta=0.5 \
	--config_override estimation_params.zeta=-0.3 \
	--config_override estimation_params.psi=0.5

echo "Finished generating all datasets."
