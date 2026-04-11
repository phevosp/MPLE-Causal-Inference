#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MANIFEST_PATH="${EXPERIMENT_MANIFEST:-$SCRIPT_DIR/experiments/SyntheticExperimentsGrid/latest_manifest.txt}"
USCOUNTY_OUTPUT_ROOT="${USCOUNTY_OUTPUT_ROOT:-experiments/USCountyVaccination_US}"
BASE_CONFIG="base_config.yaml"
OUTCOME_CODE="death_rate_100k_ge_2"
LAG_CODE="2w"
TRIM_SCOPE="trimmed"
NETWORK_NAME="contiguity"

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

shared_panel_dir() {
	local intervention_code="$1"
	printf '%s\n' \
		"$USCOUNTY_OUTPUT_ROOT/shared_panels/outcome_${OUTCOME_CODE}__intervention_${intervention_code}__lag_${LAG_CODE}__scope_${TRIM_SCOPE}"
}

ensure_uscounty_shared_panels() {
	local missing=0
	for intervention_code in complete_cov_ge_20 complete_cov_ge_30 complete_cov_ge_40; do
		local dir
		dir="$(shared_panel_dir "$intervention_code")"
		if [[ ! -f "$dir/panel_data.npz" || ! -f "$dir/z_0.npy" ]]; then
			missing=1
			break
		fi
	done

	if [[ "$missing" -eq 0 ]]; then
		return
	fi

	"${RUNNER[@]}" data/USCountyVaccination/run_us_county_vaccination_experiments.py \
		--trim \
		--outcomes "$OUTCOME_CODE" \
		--interventions complete_cov_ge_20 complete_cov_ge_30 complete_cov_ge_40 \
		--lags "$LAG_CODE" \
		--networks "$NETWORK_NAME" \
		--output_root "$USCOUNTY_OUTPUT_ROOT"
}

panel_dims() {
	local panel_path="$1"
	PANEL_PATH="$panel_path" "${RUNNER[@]}" - <<'PY'
import os
import numpy as np

with np.load(os.environ["PANEL_PATH"], allow_pickle=False) as data:
    z = np.asarray(data["z"], dtype=float)
print(f"{z.shape[1]} {z.shape[0]}")
PY
}

fixed_tau_vector() {
	local t_steps="$1"
	local value="$2"
	T_STEPS="$t_steps" TAU_VALUE="$value" "${RUNNER[@]}" - <<'PY'
import os

t_steps = int(os.environ["T_STEPS"])
value = float(os.environ["TAU_VALUE"])
print("[" + ",".join([repr(value)] * t_steps) + "]")
PY
}

fixed_z_common_args() {
	local panel_path="$1"
	local z0_path="$2"
	local shared_panel="$3"
	local intervention_code="$4"
	local n_nodes="$5"
	local t_steps="$6"
	local seed="$7"

	printf '%s\n' \
		--config_override "global_params.N=${n_nodes}" \
		--config_override "global_params.T=${t_steps}" \
		--config_override "global_params.s=0" \
		--config_override "global_params.B=1.0" \
		--config_override "global_params.gamma_matrix_generator=erdos_renyi" \
		--config_override "global_params.gamma_matrix_params.p=0.05" \
		--config_override "global_params.x_0_generator=bernoulli" \
		--config_override "global_params.x_0_params.p=0.5" \
		--config_override "generation_params.seed=${seed}" \
		--config_override "generation_params.gibbs_sweeps=5" \
		--config_override "generation_params.intervention_mode=fixed_z" \
		--config_override "generation_params.fixed_z_source.panel_path=${panel_path}" \
		--config_override "generation_params.fixed_z_source.z0_path=${z0_path}" \
		--config_override "generation_params.fixed_z_source.shared_panel_dir=${shared_panel}" \
		--config_override "generation_params.fixed_z_source.outcome_code=${OUTCOME_CODE}" \
		--config_override "generation_params.fixed_z_source.intervention_code=${intervention_code}" \
		--config_override "generation_params.fixed_z_source.lag_code=${LAG_CODE}" \
		--config_override "generation_params.fixed_z_source.trim_scope=${TRIM_SCOPE}" \
		--config_override "estimation_params.fit_intervention_model=false" \
		--config_override "estimation_params.beta=0.35" \
		--config_override "estimation_params.xi=0.25" \
		--config_override "estimation_params.eta=0.08" \
		--config_override "estimation_params.zeta=-0.25" \
		--config_override "estimation_params.psi=0.20" \
		--config_override "global_params.basis_params.field_mode=uniform" \
		--config_override "estimation_params.field_coefs=[]"
}

run_fixed_z_uniform() {
	local intervention_code="$1"
	local field_label="$2"
	local tau_value="$3"
	local seed="$4"

	local shared_panel
	shared_panel="$(shared_panel_dir "$intervention_code")"
	local panel_path="$shared_panel/panel_data.npz"
	local z0_path="$shared_panel/z_0.npy"
	if [[ ! -f "$panel_path" || ! -f "$z0_path" ]]; then
		echo "Missing fixed-z artifacts for ${intervention_code}: ${shared_panel}" >&2
		exit 1
	fi

	local dims n_nodes t_steps tau_vector
	dims="$(panel_dims "$panel_path")"
	read -r n_nodes t_steps <<<"$dims"
	tau_vector="$(fixed_tau_vector "$t_steps" "$tau_value")"

	local label="fixed_z_${intervention_code}_lag${LAG_CODE}_${TRIM_SCOPE}_uniform_${field_label}"
	mapfile -t args < <(
		fixed_z_common_args \
			"$panel_path" \
			"$z0_path" \
			"$shared_panel" \
			"$intervention_code" \
			"$n_nodes" \
			"$t_steps" \
			"$seed" \
	)
	run_generation "$label" \
		"${args[@]}" \
		--config_override "estimation_params.tau_params.mode=fixed" \
		--config_override "estimation_params.tau_params.vector=${tau_vector}"
}

run_fixed_z_latent() {
	local intervention_code="$1"
	local field_label="$2"
    local rank="$3"
    local B="$4"
	local seed="$4"

	local shared_panel
	shared_panel="$(shared_panel_dir "$intervention_code")"
	local panel_path="$shared_panel/panel_data.npz"
	local z0_path="$shared_panel/z_0.npy"
	if [[ ! -f "$panel_path" || ! -f "$z0_path" ]]; then
		echo "Missing fixed-z artifacts for ${intervention_code}: ${shared_panel}" >&2
		exit 1
	fi

	local dims n_nodes t_steps
	dims="$(panel_dims "$panel_path")"
	read -r n_nodes t_steps <<<"$dims"

	local label="fixed_z_${intervention_code}_lag${LAG_CODE}_${TRIM_SCOPE}_latent_rank${rank}_B${B}"
	mapfile -t args < <(
		fixed_z_common_args \
			"$panel_path" \
			"$z0_path" \
			"$shared_panel" \
			"$intervention_code" \
			"$n_nodes" \
			"$t_steps" \
			"$seed" \
	)
	run_generation "$label" \
		"${args[@]}" \
		--config_override "global_params.basis_params.field_mode=latent_feature_matrix" \
        --config_override "global_params.basis_params.latent_rank=${rank}" \
        --config_override "global_params.B=${B}"
}

ensure_uscounty_shared_panels

# run_fixed_z_uniform complete_cov_ge_20 zero 0.0 810
# run_fixed_z_uniform complete_cov_ge_20 uniform025 0.25 811
# run_fixed_z_uniform complete_cov_ge_30 zero 0.0 820
# run_fixed_z_uniform complete_cov_ge_30 uniform025 0.25 821
# run_fixed_z_uniform complete_cov_ge_40 zero 0.0 830 0.25
# run_fixed_z_uniform complete_cov_ge_40 uniform025 0.25 831

run_fixed_z_latent complete_cov_ge_30 latent_rank10_B1 10 1.0 920
run_fixed_z_latent complete_cov_ge_40 latent_rank50_B1 50 1.0 930
run_fixed_z_latent complete_cov_ge_30 latent_rank10_B3 10 3.0 920
run_fixed_z_latent complete_cov_ge_40 latent_rank50_B3 50 3.0 930

echo "Finished generating fixed-z synthetic datasets."
echo "Manifest: $MANIFEST_PATH"
