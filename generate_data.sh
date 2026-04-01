#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_CONFIG="base_config.yaml"
MANIFEST_PATH="${EXPERIMENT_MANIFEST:-$SCRIPT_DIR/experiments/latest_manifest.txt}"

mkdir -p "$(dirname "$MANIFEST_PATH")"
: >"$MANIFEST_PATH"

run_generation() {
	local label="$1"
	shift

	echo "Generating ${label}..."
	pixi run python -u data/synthetic_data_generation.py \
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

add_common_generation_args() {
	local target_name="$1"
	local n="$2"
	local t="$3"
	local seed="$4"
	add_override "$target_name" global_params.N "$n"
	add_override "$target_name" global_params.T "$t"
	add_override "$target_name" global_params.s 5
	add_override "$target_name" global_params.gamma_matrix_generator '"erdos_renyi"'
	add_override "$target_name" global_params.gamma_matrix_params.p 0.01
	add_override "$target_name" global_params.x_0_generator '"bernoulli"'
	add_override "$target_name" global_params.x_0_params.p 0.5
	add_override "$target_name" global_params.basis_params.num_shared_features 5
	add_override "$target_name" global_params.basis_params.shared_feature_seed "$seed"
	add_override "$target_name" generation_params.seed "$seed"
	add_override "$target_name" generation_params.gibbs_sweeps 10
}

add_temperature_args() {
	local target_name="$1"
	local regime="$2"
	case "$regime" in
		high_temp)
			add_override "$target_name" estimation_params.field_coefs '[0.18]'
			add_override "$target_name" estimation_params.beta 0.12
			add_override "$target_name" estimation_params.interaction_coefs '[0.15]'
			add_override "$target_name" estimation_params.eta 0.01
			add_override "$target_name" estimation_params.zeta -0.15
			add_override "$target_name" estimation_params.psi 0.05
			;;
		baseline_temp)
			add_override "$target_name" estimation_params.field_coefs '[0.5]'
			add_override "$target_name" estimation_params.beta 0.4
			add_override "$target_name" estimation_params.interaction_coefs '[0.5]'
			add_override "$target_name" estimation_params.eta 0.1
			add_override "$target_name" estimation_params.zeta -0.5
			add_override "$target_name" estimation_params.psi 0.4
			;;
		low_temp)
			add_override "$target_name" estimation_params.field_coefs '[1]'
			add_override "$target_name" estimation_params.beta 1.1
			add_override "$target_name" estimation_params.interaction_coefs '[2]'
			add_override "$target_name" estimation_params.eta 0.7
			add_override "$target_name" estimation_params.zeta -1
			add_override "$target_name" estimation_params.psi 0.4
			;;
		*)
			echo "Unknown temperature regime: $regime" >&2
			exit 1
			;;
	esac
}

add_graph_fro_args() {
	local target_name="$1"
	local regime="$2"
	case "$regime" in
		fro_small)
			add_override "$target_name" global_params.gamma_matrix_generator '"complete"'
			;;
		fro_medium)
			add_override "$target_name" global_params.gamma_matrix_generator '"erdos_renyi"'
			add_override "$target_name" global_params.gamma_matrix_params.p 0.10
			;;
		fro_large)
			add_override "$target_name" global_params.gamma_matrix_generator '"cycle"'
			;;
		*)
			echo "Unknown Frobenius regime: $regime" >&2
			exit 1
			;;
	esac
}

graph_family_for_regime() {
	local regime="$1"
	case "$regime" in
		fro_small)
			echo complete
			;;
		fro_medium)
			echo erdos_renyi
			;;
		fro_large)
			echo cycle
			;;
		*)
			echo unknown
			;;
	esac
}

add_simple_model_args() {
	local target_name="$1"
	add_override "$target_name" global_params.basis_params.field_mode '"uniform"'
	add_override "$target_name" global_params.basis_params.interaction_mode '"known_graph"'
}

add_shared_field_args() {
	local target_name="$1"
	local regime="$2"
	add_override "$target_name" global_params.basis_params.field_mode '"shared_feature_field"'
	case "$regime" in
		high_temp)
			add_override "$target_name" estimation_params.field_coefs '[0.175,0.140,-0.125,0.007,0.125,0.140,-0.125,-0.040,0.125,-0.180,0.185]'
			;;
		baseline_temp)
			add_override "$target_name" estimation_params.field_coefs '[0.315,0.308,0.350,-0.58,0.5,-0.6,-0.05,0.08,+0.10,0.3,0.4]'
			;;
		low_temp)
			add_override "$target_name" estimation_params.field_coefs '[0.425,0.220,0.675,0.920,-0.750,0.440,-0.175,0.5,-0.075,0.240,0.855]'
			;;
		*)
			echo "Unknown temperature regime: $regime" >&2
			exit 1
			;;
	esac
}

add_shared_interaction_args() {
	local target_name="$1"
	local regime="$2"
	add_override "$target_name" global_params.basis_params.interaction_mode '"shared_feature_interactions"'
	case "$regime" in
		high_temp)
			add_override "$target_name" estimation_params.interaction_coefs '[0.125,0.040,0.020,0.040,0.020,0.040,0.020,0.040,0.020,0.065,0.070]'
			;;
		baseline_temp)
			add_override "$target_name" estimation_params.interaction_coefs '[0.25,0.08,0.04,0.08,0.04,0.08,0.04,0.08,0.04,0.13,0.14]'
			;;
		low_temp)
			add_override "$target_name" estimation_params.interaction_coefs '[0.375,0.120,0.060,0.120,0.060,0.120,0.060,0.120,0.060,0.195,0.210]'
			;;
		*)
			echo "Unknown temperature regime: $regime" >&2
			exit 1
			;;
	esac
}

seed_counter=200

for n in 100 1000 5000; do
	for t in 10 100; do
		for temperature in high_temp baseline_temp low_temp; do
			for fro in fro_small fro_medium fro_large; do
					for field_complexity in uniform shared_feature_field; do
						for interaction_complexity in known_graph shared_feature_interactions; do
							label="N${n}_T${t}_${temperature}_${fro}_${field_complexity}_${interaction_complexity}"
							args=()
							add_metadata args suite core_grid
							add_metadata args N_regime "N${n}"
							add_metadata args T_regime "T${t}"
							add_metadata args temperature_regime "$temperature"
							add_metadata args fro_regime "$fro"
							add_metadata args graph_family "$(graph_family_for_regime "$fro")"
							add_metadata args field_complexity "$field_complexity"
							add_metadata args interaction_complexity "$interaction_complexity"
							add_common_generation_args args "$n" "$t" "$seed_counter"
							add_temperature_args args "$temperature"
							add_graph_fro_args args "$fro"
							add_simple_model_args args
							if [ "$field_complexity" = "shared_feature_field" ]; then
								add_shared_field_args args "$temperature"
							fi
							if [ "$interaction_complexity" = "shared_feature_interactions" ]; then
								add_shared_interaction_args args "$temperature"
							fi
							run_generation "$label" "${args[@]}"
							seed_counter=$((seed_counter + 1))
						done
					done
				done
			done
		done
	done
done

echo "Finished generating all datasets."
echo "Manifest: $MANIFEST_PATH"
