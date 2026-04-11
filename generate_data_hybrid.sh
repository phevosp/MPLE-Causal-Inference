#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MANIFEST_PATH="${EXPERIMENT_MANIFEST:-$SCRIPT_DIR/experiments/SyntheticHybridExperiments/latest_manifest.txt}"
EXPERIMENT_ROOT="${HYBRID_EXPERIMENT_ROOT:-experiments/SyntheticHybridExperiments}"
USCOUNTY_OUTPUT_ROOT="${USCOUNTY_OUTPUT_ROOT:-experiments/USCountyVaccination_US}"
BASE_CONFIG="base_config.yaml"
ANCHOR_OUTCOME_CODE="death_rate_100k_ge_2"
ANCHOR_INTERVENTION_CODE="${HYBRID_INTERVENTION_CODE:-complete_cov_ge_20}"
LAG_CODE="${HYBRID_LAG_CODE:-2w}"
TRIM_SCOPE="trimmed"
NETWORK_NAME="${HYBRID_NETWORK_NAME:-contiguity}"

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
		--experiment_root "$EXPERIMENT_ROOT" \
		--manifest_path "$MANIFEST_PATH" \
		"$@"
	sleep 1
}

clamp_to_bound() {
	local value="$1"
	local bound="$2"
	awk -v v="$value" -v b="$bound" 'BEGIN { if (v > b) v = b; if (v < -b) v = -b; printf "%.12g", v }'
}

realized_intervention_dir() {
	printf '%s\n' \
		"$USCOUNTY_OUTPUT_ROOT/realized_interventions/intervention_${ANCHOR_INTERVENTION_CODE}__lag_${LAG_CODE}__scope_${TRIM_SCOPE}"
}

realized_network_dir() {
	printf '%s\n' \
		"$USCOUNTY_OUTPUT_ROOT/realized_networks/network_${NETWORK_NAME}__scope_${TRIM_SCOPE}"
}

ensure_uscounty_component_artifacts() {
	local intervention_dir network_dir
	intervention_dir="$(realized_intervention_dir)"
	network_dir="$(realized_network_dir)"

	if [[ -f "$intervention_dir/panel_data.npz" && -f "$intervention_dir/z_0.npy" && -f "$network_dir/gamma_matrix_sparse.npz" ]]; then
		return
	fi

	"${RUNNER[@]}" data/USCountyVaccination/run_us_county_vaccination_experiments.py \
		--trim \
		--outcomes "$ANCHOR_OUTCOME_CODE" \
		--interventions "$ANCHOR_INTERVENTION_CODE" \
		--lags "$LAG_CODE" \
		--networks "$NETWORK_NAME" \
		--output_root "$USCOUNTY_OUTPUT_ROOT"
}

anchor_panel_dims() {
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
	T_STEPS="$t_steps" "${RUNNER[@]}" - <<'PY'
import os

t_steps = int(os.environ["T_STEPS"])
print("[" + ",".join(["0.0"] * t_steps) + "]")
PY
}

common_args() {
	local n_nodes="$1"
	local t_steps="$2"
	local seed="$3"
	local B="2.0"
	local beta eta zeta psi xi

	beta="$(clamp_to_bound "0.35" "$B")"
	eta="$(clamp_to_bound "0.08" "$B")"
	zeta="$(clamp_to_bound "-0.25" "$B")"
	psi="$(clamp_to_bound "0.20" "$B")"
	xi="$(clamp_to_bound "0.25" "$B")"

	printf '%s\n' \
		--config_override "global_params.N=${n_nodes}" \
		--config_override "global_params.T=${t_steps}" \
		--config_override "global_params.s=0" \
		--config_override "global_params.B=${B}" \
		--config_override "global_params.x_0_generator=bernoulli" \
		--config_override "global_params.x_0_params.p=0.5" \
		--config_override "generation_params.seed=${seed}" \
		--config_override "generation_params.gibbs_sweeps=5" \
		--config_override "estimation_params.fit_intervention_model=false" \
		--config_override "estimation_params.beta=${beta}" \
		--config_override "estimation_params.xi=${xi}" \
		--config_override "estimation_params.eta=${eta}" \
		--config_override "estimation_params.zeta=${zeta}" \
		--config_override "estimation_params.psi=${psi}"
}

intervention_source_args() {
	local intervention_source="$1"
	local intervention_dir="$2"

	case "$intervention_source" in
		uscounty)
			printf '%s\n' \
				--config_override "generation_params.intervention_mode=fixed_z" \
				--config_override "generation_params.fixed_z_source.panel_path=${intervention_dir}/panel_data.npz" \
				--config_override "generation_params.fixed_z_source.z0_path=${intervention_dir}/z_0.npy" \
				--config_override "generation_params.fixed_z_source.artifact_dir=${intervention_dir}" \
				--config_override "generation_params.fixed_z_source.outcome_code=${ANCHOR_OUTCOME_CODE}" \
				--config_override "generation_params.fixed_z_source.intervention_code=${ANCHOR_INTERVENTION_CODE}" \
				--config_override "generation_params.fixed_z_source.lag_code=${LAG_CODE}" \
				--config_override "generation_params.fixed_z_source.trim_scope=${TRIM_SCOPE}"
			;;
		generated)
			printf '%s\n' \
				--config_override "generation_params.intervention_mode=generated_z"
			;;
		*)
			echo "Unknown intervention source: ${intervention_source}" >&2
			exit 1
			;;
	esac
}

network_source_args() {
	local network_source="$1"
	local network_dir="$2"

	case "$network_source" in
		uscounty)
			printf '%s\n' \
				--config_override "global_params.gamma_matrix_generator=fixed_artifact" \
				--config_override "global_params.fixed_gamma_source.gamma_path=${network_dir}/gamma_matrix_sparse.npz" \
				--config_override "global_params.fixed_gamma_source.node_index_path=${network_dir}/node_index.csv" \
				--config_override "global_params.fixed_gamma_source.artifact_dir=${network_dir}" \
				--config_override "global_params.fixed_gamma_source.network_name=${NETWORK_NAME}" \
				--config_override "global_params.fixed_gamma_source.trim_scope=${TRIM_SCOPE}"
			;;
		generated)
			printf '%s\n' \
				--config_override "global_params.gamma_matrix_generator=erdos_renyi" \
				--config_override "global_params.gamma_matrix_params.p=0.05"
			;;
		*)
			echo "Unknown network source: ${network_source}" >&2
			exit 1
			;;
	esac
}

field_source_args() {
	local field_mode="$1"
	local tau_vector="$2"

	case "$field_mode" in
		zero)
			printf '%s\n' \
				--config_override "global_params.basis_params.field_mode=uniform" \
				--config_override "estimation_params.field_coefs=[]" \
				--config_override "estimation_params.tau_params.mode=fixed" \
				--config_override "estimation_params.tau_params.vector=${tau_vector}"
			;;
		latent)
			printf '%s\n' \
				--config_override "global_params.basis_params.field_mode=latent_feature_matrix" \
				--config_override "global_params.basis_params.latent_rank=40"
			;;
		*)
			echo "Unknown field mode: ${field_mode}" >&2
			exit 1
			;;
	esac
}

run_hybrid_experiment() {
	local intervention_source="$1"
	local network_source="$2"
	local field_mode="$3"
	local seed="$4"

	local intervention_dir network_dir panel_path dims n_nodes t_steps tau_vector
	intervention_dir="$(realized_intervention_dir)"
	network_dir="$(realized_network_dir)"
	panel_path="$intervention_dir/panel_data.npz"

	if [[ ! -f "$panel_path" ]]; then
		echo "Missing anchor intervention artifact: ${panel_path}" >&2
		exit 1
	fi
	if [[ "$network_source" == "uscounty" && ! -f "$network_dir/gamma_matrix_sparse.npz" ]]; then
		echo "Missing network artifact: ${network_dir}" >&2
		exit 1
	fi

	dims="$(anchor_panel_dims "$panel_path")"
	read -r n_nodes t_steps <<<"$dims"
	tau_vector="$(fixed_tau_vector "$t_steps")"

	local label="hybrid_${intervention_source}_interv_${network_source}_net_${field_mode}"
	mapfile -t args < <(
		common_args "$n_nodes" "$t_steps" "$seed"
		intervention_source_args "$intervention_source" "$intervention_dir"
		network_source_args "$network_source" "$network_dir"
		field_source_args "$field_mode" "$tau_vector"
	)

	run_generation "$label" \
		"${args[@]}" \
		--metadata "hybrid_mode=uscounty_generated_sources" \
		--metadata "intervention_source=${intervention_source}" \
		--metadata "network_source=${network_source}" \
		--metadata "outcome_field_setting=${field_mode}" \
		--metadata "anchor_outcome_code=${ANCHOR_OUTCOME_CODE}" \
		--metadata "anchor_intervention_code=${ANCHOR_INTERVENTION_CODE}" \
		--metadata "anchor_lag_code=${LAG_CODE}" \
		--metadata "anchor_network_name=${NETWORK_NAME}" \
		--metadata "trim_scope=${TRIM_SCOPE}"
}

ensure_uscounty_component_artifacts

for intervention_source in uscounty generated; do
	for network_source in uscounty generated; do
		if [[ "$intervention_source" == "generated" && "$network_source" == "generated" ]]; then
			continue
		fi
		for field_mode in zero latent; do
			seed_value=$((900 + ${#intervention_source} + ${#network_source} + ${#field_mode}))
			run_hybrid_experiment "$intervention_source" "$network_source" "$field_mode" "$seed_value"
		done
	done
done

echo "Finished generating hybrid synthetic datasets."
echo "Manifest: $MANIFEST_PATH"
