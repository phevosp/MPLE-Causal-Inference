#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MANIFEST_PATH="${EXPERIMENT_MANIFEST:-$SCRIPT_DIR/experiments/SyntheticExperimentsGrid/latest_manifest.txt}"
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

clamp_to_bound() {
	local value="$1"
	local bound="$2"
	awk -v v="$value" -v b="$bound" 'BEGIN { if (v > b) v = b; if (v < -b) v = -b; printf "%.12g", v }'
}

common_args() {
	local n="$1"
	local t="$2"
	local s="$3"
	local seed="$4"
	local xi="$5"
	local B="2.0"
	local beta eta zeta psi xi_clamped

	beta="$(clamp_to_bound "0.35" "$B")"
	eta="$(clamp_to_bound "0.08" "$B")"
	zeta="$(clamp_to_bound "-0.25" "$B")"
	psi="$(clamp_to_bound "0.20" "$B")"
	xi_clamped="$(clamp_to_bound "$xi" "$B")"

	printf '%s\n' \
		--config_override "global_params.N=${n}" \
		--config_override "global_params.T=${t}" \
		--config_override "global_params.s=${s}" \
		--config_override "global_params.B=${B}" \
		--config_override "global_params.gamma_matrix_generator=erdos_renyi" \
		--config_override "global_params.gamma_matrix_params.p=0.05" \
		--config_override "global_params.x_0_generator=bernoulli" \
		--config_override "global_params.x_0_params.p=0.5" \
		--config_override "generation_params.seed=${seed}" \
		--config_override "generation_params.gibbs_sweeps=5" \
		--config_override "generation_params.intervention_mode=generated_z" \
		--config_override "estimation_params.beta=${beta}" \
		--config_override "estimation_params.xi=${xi_clamped}" \
		--config_override "estimation_params.eta=${eta}" \
		--config_override "estimation_params.zeta=${zeta}" \
		--config_override "estimation_params.psi=${psi}"
}

run_uniform() {
	local n="$1"
	local t="$2"
	local s="$3"
	local xi="$4"
	local seed="$5"
	local xi_slug="${xi/./p}"
	local label="generated_z_uniform_n${n}_t${t}_xi${xi_slug}"
	mapfile -t args < <(common_args "$n" "$t" "$s" "$seed" "$xi")
	run_generation "$label" \
		"${args[@]}" \
		--config_override "global_params.basis_params.field_mode=uniform" \
		--config_override "estimation_params.field_coefs=[]" \
		--config_override "estimation_params.tau_params.mode=uniform_random" \
		--config_override "estimation_params.tau_params.lower=-0.20" \
		--config_override "estimation_params.tau_params.upper=0.20" \
		--config_override "estimation_params.tau_params.seed=${seed}"
}

run_shared() {
	local n="$1"
	local t="$2"
	local s="$3"
	local xi="$4"
	local seed="$5"
	local xi_slug="${xi/./p}"
	local label="generated_z_shared_n${n}_t${t}_xi${xi_slug}"
	mapfile -t args < <(common_args "$n" "$t" "$s" "$seed" "$xi")
	run_generation "$label" \
		"${args[@]}" \
		--config_override "global_params.basis_params.field_mode=shared_feature_field" \
		--config_override "global_params.basis_params.num_shared_features=3" \
		--config_override "global_params.basis_params.shared_feature_seed=${seed}" \
		--config_override "estimation_params.field_coefs=[0.20,-0.10,0.15,-0.08,0.12,-0.05]" \
		--config_override "estimation_params.tau_params.mode=uniform_random" \
		--config_override "estimation_params.tau_params.lower=-0.20" \
		--config_override "estimation_params.tau_params.upper=0.20" \
		--config_override "estimation_params.tau_params.seed=${seed}"
}

run_latent() {
	local n="$1"
	local t="$2"
	local s="$3"
	local xi="$4"
	local B="$5"
	local rank="$6"
	local seed="$7"
	local beta eta zeta psi xi_clamped
	beta="$(clamp_to_bound "0.35" "$B")"
	eta="$(clamp_to_bound "0.08" "$B")"
	zeta="$(clamp_to_bound "-0.25" "$B")"
	psi="$(clamp_to_bound "0.20" "$B")"
	xi_clamped="$(clamp_to_bound "$xi" "$B")"
	local xi_slug="${xi_clamped/./p}"
	local label="generated_z_latent_n${n}_t${t}_xi${xi_slug}_B${B}_rank${rank}"
	mapfile -t args < <(common_args "$n" "$t" "$s" "$seed" "$xi")
	run_generation "$label" \
		"${args[@]}" \
		--config_override "global_params.basis_params.field_mode=latent_feature_matrix" \
		--config_override "global_params.basis_params.latent_rank=${rank}" \
		--config_override "global_params.B=${B}" \
		--config_override "estimation_params.beta=${beta}" \
		--config_override "estimation_params.xi=${xi_clamped}" \
		--config_override "estimation_params.eta=${eta}" \
		--config_override "estimation_params.zeta=${zeta}" \
		--config_override "estimation_params.psi=${psi}"
}

# Compact generated-z sweep covering all active field modes, all target xi values,
# and small/medium/larger panel sizes without running the full cross product.
run_uniform 100 20 12 0.25 700
run_uniform 300 40 20 0.75 701
run_uniform 600 60 30 1.5 702

# run_shared 100 20 12 0.25 710
# run_shared 300 40 20 0.75 711
# run_shared 600 60 30 1.5 712

run_latent 100 20 12 0.25 1 4 720
run_latent 300 40 20 0.75 1 6 721
run_latent 600 60 30 1.5 1 8 722
run_latent 600 60 12 0.75 2 6 723
run_latent 1000 60 20 0.75 4 10 724

echo "Finished generating synthetic datasets."
echo "Manifest: $MANIFEST_PATH"
