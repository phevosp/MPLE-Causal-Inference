from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from pipeline_specs import (
    expand_named_entries,
    read_csv_manifest,
    slugify,
    write_csv_manifest,
)
from posterior_predictive_utils import (
    COUNTERFACTUAL_MANIFEST_NAME,
    COUNTERFACTUAL_ROOT_NAME,
    _io_path,
    compute_panel_statistics,
    compute_counterfactual_sample_summary,
    load_experiment_panel_context,
    load_fit_parameter_bundle,
    load_truth_parameter_bundle,
    resolve_intervention_context,
    simulate_outcomes_for_bundle,
    summarize_predictive_statistics,
    write_counterfactual_summary_tables,
    write_predictive_stats_tables,
)
from report_posterior_predictive import write_posterior_predictive_reports


POSTERIOR_PREDICTIVE_ROOT_NAME = "posterior_predictive"
POSTERIOR_PREDICTIVE_MANIFEST_NAME = "posterior_predictive_manifest.csv"


def _index_generation_rows(
    generation_manifest_path: str | Path,
) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in read_csv_manifest(generation_manifest_path):
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


def _resolve_fit_lookup(
    fit_manifest_path: str | Path,
) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv_manifest(fit_manifest_path):
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


def _as_bool(value: object, default: bool = True) -> bool:
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


def _experiment_has_truth(experiment_row: dict[str, str]) -> bool:
    if "has_truth" in experiment_row:
        return _as_bool(experiment_row.get("has_truth"), default=True)
    metadata_path = (
        Path(str(experiment_row.get("experiment_path", "")))
        / "experiment_metadata.yaml"
    )
    if not metadata_path.exists():
        return True
    with open(_io_path(metadata_path), "r", encoding="utf-8") as handle:
        metadata = OmegaConf.to_container(OmegaConf.load(handle), resolve=True)
    if not isinstance(metadata, dict):
        return True
    return _as_bool(metadata.get("has_truth"), default=True)


def _resolve_target_pairs(
    target_pairs_path: str | Path,
    generation_lookup: dict[str, dict[str, str]],
    fit_lookup: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
    resolved_targets: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in read_csv_manifest(target_pairs_path):
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
            if not _experiment_has_truth(experiment_row):
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
                f"Target-pairs manifest {target_pairs_path} has invalid source_type '{source_type}' for experiment '{experiment_name}'."
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
                "source_type": source_type,
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


def _simulate_target(
    target: dict[str, object],
    run_spec: dict[str, object],
    overwrite: bool,
) -> dict[str, object]:
    experiment_row = target["experiment_row"]
    experiment_root = Path(str(experiment_row["experiment_path"])).resolve()
    run_slug = str(run_spec["slug"])
    source_slug = str(target["source_slug"])
    intervention_source = str(target["intervention_source"])
    intervention_name = str(target["intervention_name"])
    intervention_slug = str(target["intervention_slug"])
    if intervention_source == "observed_experiment":
        output_root = (
            experiment_root / POSTERIOR_PREDICTIVE_ROOT_NAME / source_slug / run_slug
        )
    else:
        output_root = (
            experiment_root
            / COUNTERFACTUAL_ROOT_NAME
            / source_slug
            / intervention_slug
            / run_slug
        )
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"{output_root} already exists. Re-run with --overwrite to rebuild it."
            )
    output_root.mkdir(parents=True, exist_ok=False)

    panel_context = load_experiment_panel_context(experiment_root)
    intervention_context = resolve_intervention_context(
        experiment_root,
        intervention_source=intervention_source,
        intervention_name=intervention_name,
        panel_context=panel_context,
    )
    if target["source_type"] == "truth":
        bundle = load_truth_parameter_bundle(experiment_root)
    else:
        fit_row = target["fit_row"]
        fit_root = Path(str(fit_row["fit_path"]))
        bundle = load_fit_parameter_bundle(fit_root, experiment_root)
    if int(bundle.t_steps) != int(panel_context["T"]):
        raise ValueError(
            f"Posterior-predictive source '{target['source_name']}' has t_steps={bundle.t_steps},"
            f" but experiment '{experiment_row.get('experiment_name', experiment_root.name)}' has T={panel_context['T']}."
        )
    num_samples = int(run_spec["num_samples"])
    gibbs_sweeps = int(run_spec["gibbs_sweeps"])
    seed = int(run_spec["seed"])
    simulated_stats: list[dict[str, float | None]] = []
    counterfactual_sample_summaries: dict[str, list[object]] = {
        "overall_mean_magnetization": [],
        "post_intervention_mean_magnetization": [],
        "unit_mean_magnetization": [],
    }
    observed_stats = None
    if intervention_source == "observed_experiment":
        observed_stats = compute_panel_statistics(
            panel_context["x"],
            z=intervention_context.z,
            x_0=panel_context["x_0"],
            s=int(intervention_context.s),
            field_matrix=bundle.field_matrix,
            gamma_matrix=bundle.gamma_matrix,
        )
    for sample_index in range(num_samples):
        print(
            f"Simulating posterior-predictive sample {sample_index + 1} / {num_samples} for target '{target['source_name']}', intervention '{intervention_name}', and run '{run_spec['name']}'..."
        )
        sample_x = simulate_outcomes_for_bundle(
            bundle,
            x_0=panel_context["x_0"],
            z=intervention_context.z,
            gibbs_sweeps=gibbs_sweeps,
            seed=seed + sample_index,
        )
        if intervention_source == "observed_experiment":
            simulated_stats.append(
                compute_panel_statistics(
                    sample_x,
                    z=intervention_context.z,
                    x_0=panel_context["x_0"],
                    s=int(intervention_context.s),
                    field_matrix=bundle.field_matrix,
                    gamma_matrix=bundle.gamma_matrix,
                )
            )
        else:
            sample_summary = compute_counterfactual_sample_summary(
                sample_x,
                s=int(intervention_context.s),
            )
            counterfactual_sample_summaries["overall_mean_magnetization"].append(
                sample_summary["overall_mean_magnetization"]
            )
            counterfactual_sample_summaries[
                "post_intervention_mean_magnetization"
            ].append(sample_summary["post_intervention_mean_magnetization"])
            counterfactual_sample_summaries["unit_mean_magnetization"].append(
                sample_summary["unit_mean_magnetization"]
            )

    summary: dict[str, float | int | str]
    if intervention_source == "observed_experiment":
        stat_rows, summary = summarize_predictive_statistics(
            observed_stats, simulated_stats
        )
        write_predictive_stats_tables(output_root, stat_rows)
    else:
        sample_summaries = {
            "overall_mean_magnetization": np.asarray(
                counterfactual_sample_summaries["overall_mean_magnetization"],
                dtype=float,
            ),
            "post_intervention_mean_magnetization": np.asarray(
                counterfactual_sample_summaries["post_intervention_mean_magnetization"],
                dtype=float,
            ),
            "unit_mean_magnetization": np.asarray(
                counterfactual_sample_summaries["unit_mean_magnetization"],
                dtype=float,
            ),
        }
        write_counterfactual_summary_tables(
            output_root,
            sample_summaries=sample_summaries,
        )
        summary = {
            "num_samples": int(num_samples),
            "num_units": int(panel_context["N"]),
        }
    metadata = {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_path": str(experiment_root),
        "run_name": run_spec["name"],
        "run_slug": run_slug,
        "source_type": target["source_type"],
        "source_name": target["source_name"],
        "source_slug": source_slug,
        "intervention_source": intervention_source,
        "intervention_name": intervention_name,
        "intervention_slug": intervention_slug,
        "latent_rank": int(bundle.latent_rank),
        "num_samples": num_samples,
        "gibbs_sweeps": gibbs_sweeps,
        "seed": seed,
        "s": int(intervention_context.s),
        "summary": summary,
    }
    metadata_path = output_root / (
        "posterior_predictive_metadata.yaml"
        if intervention_source == "observed_experiment"
        else "counterfactual_metadata.yaml"
    )
    with open(_io_path(metadata_path), "w", encoding="utf-8") as handle:
        OmegaConf.save(OmegaConf.create(metadata), handle)
    base_row = {
        "experiment_name": experiment_row.get("experiment_name", ""),
        "experiment_slug": experiment_row.get("experiment_slug", ""),
        "descriptor": experiment_row.get("descriptor", ""),
        "experiment_path": str(experiment_root),
        "intervention_source": experiment_row.get("intervention_source", ""),
        "graph_source": experiment_row.get("graph_source", ""),
        "N": panel_context["N"],
        "T": panel_context["T"],
        "s": int(intervention_context.s),
        "run_name": run_spec["name"],
        "run_slug": run_slug,
        "source_type": target["source_type"],
        "source_name": target["source_name"],
        "source_slug": source_slug,
        "target_intervention_source": intervention_source,
        "target_intervention_name": intervention_name,
        "target_intervention_slug": intervention_slug,
        "latent_rank": int(bundle.latent_rank),
        "num_samples": num_samples,
        "gibbs_sweeps": gibbs_sweeps,
        "seed": seed,
        "output_path": str(output_root),
    }
    if intervention_source == "observed_experiment":
        return {
            **base_row,
            "mean_abs_zscore": float(summary["mean_abs_zscore"]),
            "max_abs_zscore": float(summary["max_abs_zscore"]),
            "coverage_rate": float(summary["coverage_rate"]),
            "num_statistics": int(summary["num_statistics"]),
        }
    return {
        **base_row,
        "intervention_source": intervention_source,
        "intervention_name": intervention_name,
        "intervention_slug": intervention_slug,
    }


def run_posterior_predictive(
    generation_manifest_path: str | Path,
    fit_manifest_path: str | Path,
    target_pairs_path: str | Path,
    spec_path: str | Path,
    overwrite: bool = False,
) -> Path:
    generation_lookup = _index_generation_rows(generation_manifest_path)
    fit_lookup = _resolve_fit_lookup(fit_manifest_path)
    targets = _resolve_target_pairs(target_pairs_path, generation_lookup, fit_lookup)
    runs = expand_named_entries(spec_path, "runs")
    if not runs:
        raise ValueError(f"No runs found in posterior predictive spec {spec_path}.")

    predictive_rows: list[dict[str, object]] = []
    counterfactual_rows: list[dict[str, object]] = []
    for target in targets:
        for run_spec in runs:
            row = _simulate_target(
                target,
                run_spec,
                overwrite=overwrite,
            )
            if str(target["intervention_source"]) == "observed_experiment":
                predictive_rows.append(row)
            else:
                counterfactual_rows.append(row)

    manifest_root = Path(generation_manifest_path).resolve().parent
    predictive_manifest_path = manifest_root / POSTERIOR_PREDICTIVE_MANIFEST_NAME
    counterfactual_manifest_path = manifest_root / COUNTERFACTUAL_MANIFEST_NAME
    if predictive_rows:
        write_csv_manifest(predictive_manifest_path, predictive_rows)
        write_posterior_predictive_reports(predictive_manifest_path)
    if counterfactual_rows:
        write_csv_manifest(counterfactual_manifest_path, counterfactual_rows)
    if predictive_rows:
        return predictive_manifest_path
    if counterfactual_rows:
        return counterfactual_manifest_path
    raise ValueError("No posterior-predictive or counterfactual runs were produced.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run posterior-predictive outcome simulations over explicit experiment/source targets."
    )
    parser.add_argument("--generation_manifest_path", required=True, type=str)
    parser.add_argument("--fit_manifest_path", required=True, type=str)
    parser.add_argument("--target_pairs_path", required=True, type=str)
    parser.add_argument(
        "--spec_path",
        type=str,
        default="data/configs/posterior_predictive_spec.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = run_posterior_predictive(
        args.generation_manifest_path,
        args.fit_manifest_path,
        args.target_pairs_path,
        args.spec_path,
        overwrite=args.overwrite,
    )
    print(f"Posterior predictive manifest: {manifest_path}")


if __name__ == "__main__":
    main()
