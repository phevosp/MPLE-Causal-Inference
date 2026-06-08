"""Posterior predictive generation and fit manifest management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.t0_config_utils import load_yaml_mapping
from utils.t0_csv_utils import read_csv_rows
from utils.t0_string_utils import slugify
from utils.t6_pipeline_spec_utils import expand_named_entries


POSTERIOR_PREDICTIVE_ROOT_NAME = "posterior_predictive"
POSTERIOR_PREDICTIVE_MANIFEST_NAME = "posterior_predictive_manifest.csv"


def as_bool(value: object, default: bool = True) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def experiment_has_truth(experiment_row: dict[str, str]) -> bool:
    if "has_truth" in experiment_row:
        return as_bool(experiment_row.get("has_truth"), default=True)
    metadata_path = (
        Path(str(experiment_row.get("experiment_path", "")))
        / "experiment_metadata.yaml"
    )
    if not metadata_path.exists():
        return True
    metadata = load_yaml_mapping(metadata_path)
    return as_bool(metadata.get("has_truth"), default=True)


def index_generation_rows(
    generation_manifest_path: str | Path,
) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(generation_manifest_path):
        experiment_name = str(row.get("experiment_name", "")).strip()
        if not experiment_name:
            raise ValueError(
                f"Generation manifest {generation_manifest_path} contains a row without experiment_name."
            )
        if experiment_name in index:
            raise ValueError(
                f"Generation manifest {generation_manifest_path} contains duplicate experiment_name '{experiment_name}'."
            )
        index[experiment_name] = row
    return index


def resolve_fit_lookup(
    fit_manifest_path: str | Path,
) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv_rows(fit_manifest_path):
        experiment_name = str(row.get("experiment_name", "")).strip()
        variant_name = str(row.get("variant_name", "")).strip()
        if not experiment_name or not variant_name:
            raise ValueError(
                f"Fit manifest {fit_manifest_path} must provide experiment_name and variant_name on every row."
            )
        key = (experiment_name, variant_name)
        if key in lookup:
            raise ValueError(
                f"Fit manifest {fit_manifest_path} contains duplicate fit row for experiment '{experiment_name}' and variant '{variant_name}'."
            )
        lookup[key] = row
    return lookup


def resolve_target_pairs(
    target_pairs_path: str | Path,
    generation_lookup: dict[str, dict[str, str]],
    fit_lookup: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
    resolved_targets: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for row in read_csv_rows(target_pairs_path):
        experiment_name = str(row.get("experiment_name", "")).strip()
        source_type = str(row.get("source_type", "")).strip().lower()
        variant_name = str(row.get("variant_name", "")).strip()
        intervention_source = (
            str(row.get("intervention_source", "observed_experiment")).strip().lower()
            or "observed_experiment"
        )
        intervention_name = str(row.get("intervention_name", "")).strip()
        if not experiment_name:
            raise ValueError(
                f"Target-pairs manifest {target_pairs_path} contains a row without experiment_name."
            )
        if experiment_name not in generation_lookup:
            raise ValueError(
                f"Target-pairs manifest {target_pairs_path} references unknown experiment '{experiment_name}'."
            )
        experiment_row = generation_lookup[experiment_name]
        if source_type == "truth":
            if not experiment_has_truth(experiment_row):
                raise ValueError(
                    f"Truth posterior-predictive targets are not available for experiment '{experiment_name}' because has_truth=false. Use source_type='fit' instead."
                )
            if variant_name:
                raise ValueError(
                    f"Truth target for experiment '{experiment_name}' must leave variant_name blank."
                )
            source_name = "truth"
            source_slug = "truth"
            fit_row = None
        elif source_type == "truth_xi_zero":
            if not experiment_has_truth(experiment_row):
                raise ValueError(
                    f"Truth posterior-predictive targets are not available for experiment '{experiment_name}' because has_truth=false. Use source_type='fit' instead."
                )
            if variant_name:
                raise ValueError(
                    f"Truth target for experiment '{experiment_name}' must leave variant_name blank."
                )
            source_name = "truth_xi_zero"
            source_slug = "truth_xi_zero"
            fit_row = None
        elif source_type == "fit":
            if not variant_name:
                raise ValueError(
                    f"Fit target for experiment '{experiment_name}' must specify variant_name."
                )
            fit_row = fit_lookup.get((experiment_name, variant_name))
            if fit_row is None:
                raise ValueError(
                    f"Target-pairs manifest {target_pairs_path} references missing fit variant '{variant_name}' for experiment '{experiment_name}'."
                )
            source_name = str(fit_row["variant_name"])
            source_slug = f"fit_{str(fit_row['variant_slug']).strip()}"
        else:
            raise ValueError(
                f"Target-pairs manifest {target_pairs_path} has invalid source_type '{source_type}' for experiment '{experiment_name}'. Valid types are: 'truth', 'truth_xi_zero', 'fit'."
            )

        if intervention_source == "observed_experiment":
            intervention_name_value = intervention_name or "observed_experiment"
            intervention_slug = "observed_experiment"
        elif intervention_source == "saved_intervention":
            if not intervention_name:
                raise ValueError(
                    f"Saved intervention target for experiment '{experiment_name}' must specify intervention_name."
                )
            intervention_name_value = intervention_name
            intervention_slug = slugify(intervention_name_value)
        else:
            raise ValueError(
                f"Target-pairs manifest {target_pairs_path} has invalid intervention_source '{intervention_source}' for experiment '{experiment_name}'."
            )

        dedupe_key = (
            experiment_name,
            source_slug,
            intervention_source,
            intervention_slug,
        )
        if dedupe_key in seen_keys:
            raise ValueError(
                f"Target-pairs manifest {target_pairs_path} contains duplicate target for experiment '{experiment_name}', source '{source_slug}', and intervention '{intervention_slug}'."
            )
        seen_keys.add(dedupe_key)
        resolved_targets.append(
            {
                "experiment_row": experiment_row,
                "experiment_name": experiment_name,
                "source_type": source_type,
                "variant_name": variant_name,
                "source_name": source_name,
                "source_slug": source_slug,
                "fit_row": fit_row,
                "intervention_source": intervention_source,
                "intervention_name": intervention_name_value,
                "intervention_slug": intervention_slug,
            }
        )
    if not resolved_targets:
        raise ValueError(f"No target pairs found in {target_pairs_path}.")
    return resolved_targets


def resolve_run_spec(spec_path: str | Path, run_name: str) -> dict[str, Any]:
    runs = expand_named_entries(spec_path, "runs")
    if not runs:
        raise ValueError(f"No runs found in posterior predictive spec {spec_path}.")
    for run_spec in runs:
        if str(run_spec.get("name", "")).strip() == run_name:
            return run_spec
    raise ValueError(f"Run '{run_name}' was not found in posterior predictive spec {spec_path}.")


def select_target(
    targets: list[dict[str, object]],
    *,
    experiment_name: str,
    source_type: str,
    variant_name: str,
    intervention_source: str,
    intervention_name: str,
) -> dict[str, object]:
    normalized_source_type = source_type.strip().lower()
    normalized_intervention_source = intervention_source.strip().lower()
    normalized_variant_name = variant_name.strip()
    normalized_intervention_name = (
        intervention_name.strip() or "observed_experiment"
        if normalized_intervention_source == "observed_experiment"
        else intervention_name.strip()
    )
    matches = [
        target
        for target in targets
        if str(target.get("experiment_name", "")).strip() == experiment_name.strip()
        and str(target.get("source_type", "")).strip().lower() == normalized_source_type
        and str(target.get("variant_name", "")).strip() == normalized_variant_name
        and str(target.get("intervention_source", "")).strip().lower()
        == normalized_intervention_source
        and str(target.get("intervention_name", "")).strip() == normalized_intervention_name
    ]
    if not matches:
        raise ValueError(
            "No posterior-predictive target matched the requested "
            f"(experiment_name={experiment_name!r}, source_type={source_type!r}, "
            f"variant_name={variant_name!r}, intervention_source={intervention_source!r}, "
            f"intervention_name={intervention_name!r})."
        )
    if len(matches) > 1:
        raise ValueError(
            "Target selection was not unique for "
            f"(experiment_name={experiment_name!r}, source_type={source_type!r}, "
            f"variant_name={variant_name!r}, intervention_source={intervention_source!r}, "
            f"intervention_name={intervention_name!r})."
        )
    return matches[0]
