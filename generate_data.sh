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
	# synthetic_data_generation.py uses second-level timestamps for output folders
	sleep 1
}

COMMON_CONDITIONAL=(
	--config_override generation_params.process=conditional
	--config_override generation_params.gibbs_sweeps=25
)

COMMON_ISING=(
	--config_override generation_params.process=Ising
	--config_override generation_params.gibbs_sweeps=10
)

run_generation "conditional baseline" "${COMMON_CONDITIONAL[@]}"
run_generation "conditional p_large" "${COMMON_CONDITIONAL[@]}" \
	--config_override global_params.gamma_matrix_params.p=0.1
run_generation "conditional N_large" "${COMMON_CONDITIONAL[@]}" \
	--config_override global_params.N=5000
run_generation "conditional params_large" "${COMMON_CONDITIONAL[@]}" \
	--config_override estimation_params.alpha=1 \
	--config_override estimation_params.beta=1 \
	--config_override estimation_params.xi=0.5 \
	--config_override estimation_params.eta=0.1 \
	--config_override estimation_params.zeta=-0.5 \
	--config_override estimation_params.psi=0.5
run_generation "conditional robustness" "${COMMON_CONDITIONAL[@]}" \
	--config_override global_params.N=5000 \
	--config_override global_params.T=10 \
	--config_override global_params.s=0 \
	--config_override global_params.gamma_matrix_generator=complete \
	--config_override global_params.x_0_params.p=0.2 \
	--config_override estimation_params.alpha=1 \
	--config_override estimation_params.beta=-1 \
	--config_override estimation_params.xi=1 \
	--config_override estimation_params.eta=0.5 \
	--config_override estimation_params.zeta=-0.3 \
	--config_override estimation_params.psi=0.5

run_generation "Ising baseline" "${COMMON_ISING[@]}"
run_generation "Ising p_large" "${COMMON_ISING[@]}" \
	--config_override global_params.gamma_matrix_params.p=0.1
run_generation "Ising N_large" "${COMMON_ISING[@]}" \
	--config_override global_params.N=5000
run_generation "Ising params_large" "${COMMON_ISING[@]}" \
	--config_override estimation_params.alpha=1 \
	--config_override estimation_params.beta=1 \
	--config_override estimation_params.xi=0.5 \
	--config_override estimation_params.eta=0.1 \
	--config_override estimation_params.zeta=-0.5 \
	--config_override estimation_params.psi=0.5
run_generation "Ising robustness" "${COMMON_ISING[@]}" \
	--config_override global_params.N=5000 \
	--config_override global_params.T=10 \
	--config_override global_params.s=0 \
	--config_override global_params.gamma_matrix_generator=complete \
	--config_override global_params.x_0_params.p=0.2 \
	--config_override estimation_params.alpha=1 \
	--config_override estimation_params.beta=-1 \
	--config_override estimation_params.xi=1 \
	--config_override estimation_params.eta=0.5 \
	--config_override estimation_params.zeta=-0.3 \
	--config_override estimation_params.psi=0.5

echo "Finished generating all datasets."