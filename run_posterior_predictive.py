"""Run a single posterior-predictive or counterfactual target/run combination."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from intervention_utils import COUNTERFACTUAL_ROOT_NAME, resolve_intervention_context
from io_utils import io_path, write_counterfactual_summary_tables, write_predictive_stats_tables
from loading_utils import (
    load_experiment_panel_context,
    load_fit_parameter_bundle,
    load_truth_parameter_bundle,
)
from posterior_predictive_job_utils import (
    POSTERIOR_PREDICTIVE_ROOT_NAME,
    build_manifest_row,
    index_generation_rows,
    resolve_fit_lookup,
    resolve_run_spec,
    resolve_target_pairs,
    select_target,
)
from posterior_predictive_utils import (
    compute_panel_statistics,
    compute_counterfactual_sample_summary,
    simulate_outcomes_for_bundle,
    summarize_predictive_statistics,
)


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
    elif target["source_type"] == "truth_xi_zero":
        from dataclasses import replace

        bundle = load_truth_parameter_bundle(experiment_root)
        bundle = replace(bundle, xi=0.0)
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
        "time_mean_magnetization": [],
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
            s=int(intervention_context.s),
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
            counterfactual_sample_summaries["time_mean_magnetization"].append(
                sample_summary["time_mean_magnetization"]
            )

    summary: dict[str, float | int | str] = {"s": int(intervention_context.s)}
    if intervention_source == "observed_experiment":
        stat_rows, predictive_summary = summarize_predictive_statistics(
            observed_stats, simulated_stats
        )
        write_predictive_stats_tables(output_root, stat_rows)
        summary.update(predictive_summary)
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
            "time_mean_magnetization": np.asarray(
                counterfactual_sample_summaries["time_mean_magnetization"],
                dtype=float,
            ),
        }
        write_counterfactual_summary_tables(
            output_root,
            sample_summaries=sample_summaries,
        )
        summary.update(
            {
                "num_samples": int(num_samples),
                "num_units": int(panel_context["N"]),
            }
        )
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
        "num_units": int(panel_context["N"]),
        "num_time_steps": int(panel_context["T"]),
        "summary": summary,
    }
    metadata_path = output_root / (
        "posterior_predictive_metadata.yaml"
        if intervention_source == "observed_experiment"
        else "counterfactual_metadata.yaml"
    )
    with open(io_path(metadata_path), "w", encoding="utf-8") as handle:
        OmegaConf.save(OmegaConf.create(metadata), handle)
    return build_manifest_row(
        experiment_row=experiment_row,
        panel_context=panel_context,
        target=target,
        run_spec=run_spec,
        latent_rank=int(bundle.latent_rank),
        num_samples=num_samples,
        gibbs_sweeps=gibbs_sweeps,
        seed=seed,
        output_root=output_root,
        summary=summary,
    )


def run_posterior_predictive(
    generation_manifest_path: str | Path,
    fit_manifest_path: str | Path,
    target_pairs_path: str | Path,
    spec_path: str | Path,
    *,
    experiment_name: str,
    source_type: str,
    variant_name: str,
    intervention_source: str,
    intervention_name: str,
    run_name: str,
    overwrite: bool = False,
) -> dict[str, object]:
    generation_lookup = index_generation_rows(generation_manifest_path)
    fit_lookup = resolve_fit_lookup(fit_manifest_path)
    targets = resolve_target_pairs(target_pairs_path, generation_lookup, fit_lookup)
    target = select_target(
        targets,
        experiment_name=experiment_name,
        source_type=source_type,
        variant_name=variant_name,
        intervention_source=intervention_source,
        intervention_name=intervention_name,
    )
    run_spec = resolve_run_spec(spec_path, run_name)
    return _simulate_target(target, run_spec, overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one posterior-predictive outcome simulation for one explicit target and run."
    )
    parser.add_argument("--generation_manifest_path", required=True, type=str)
    parser.add_argument("--fit_manifest_path", required=True, type=str)
    parser.add_argument("--target_pairs_path", required=True, type=str)
    parser.add_argument(
        "--spec_path",
        type=str,
        default="data/configs/posterior_predictive_spec.yaml",
    )
    parser.add_argument("--experiment_name", required=True, type=str)
    parser.add_argument("--source_type", required=True, type=str)
    parser.add_argument("--variant_name", type=str, default="")
    parser.add_argument("--intervention_source", required=True, type=str)
    parser.add_argument("--intervention_name", type=str, default="")
    parser.add_argument("--run_name", required=True, type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    row = run_posterior_predictive(
        args.generation_manifest_path,
        args.fit_manifest_path,
        args.target_pairs_path,
        args.spec_path,
        experiment_name=args.experiment_name,
        source_type=args.source_type,
        variant_name=args.variant_name,
        intervention_source=args.intervention_source,
        intervention_name=args.intervention_name,
        run_name=args.run_name,
        overwrite=args.overwrite,
    )
    print(f"Posterior predictive output: {row['output_path']}")


if __name__ == "__main__":
    main()
