"""Run one posterior-predictive or counterfactual simulation target.

This file is the single-target worker used by the broader posterior-predictive
pipeline. Its job is to:

1. Resolve one explicit target from the manifests/configs.
2. Load the experiment panel plus either truth or fitted parameters.
3. Simulate repeated outcome draws under a fixed intervention panel.
4. Write the branch-specific summary artifacts for reporting.
5. Return the manifest row that the posterior-predictive reporting utilities later refresh.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from utils.t6_intervention_utils import COUNTERFACTUAL_ROOT_NAME, resolve_intervention_context
from utils.t8_output_writers import (
    write_counterfactual_summary_tables,
    write_observed_predictive_summary_tables,
    write_predictive_stats_tables,
)
from utils.t0_path_utils import io_path
from utils.t5_experiment_context import load_experiment_panel_context
from utils.t5_parameter_bundles import (
    load_fit_parameter_bundle,
    load_truth_parameter_bundle,
)
from utils.t6_posterior_predictive_manifest import (
    POSTERIOR_PREDICTIVE_ROOT_NAME,
    index_generation_rows,
    resolve_fit_lookup,
    resolve_run_spec,
    resolve_target_pairs,
    select_target,
)
from utils.t6_posterior_predictive_summary import build_manifest_row
from utils.t8_posterior_predictive_sim import (
    compute_panel_statistics,
    compute_counterfactual_sample_summary,
    simulate_outcomes_for_bundle,
)
from utils.t8_posterior_predictive_reporting import (
    summarize_observed_mean_statistics,
    summarize_predictive_statistics,
)


def _posterior_sample_seed(
    *,
    target: dict[str, object],
    run_spec: dict[str, object],
    sample_index: int,
) -> int:
    """Derive a reproducible but target-specific seed for one posterior sample."""
    base_seed = int(run_spec["seed"])
    experiment_row = target["experiment_row"]
    token = "|".join(
        [
            str(base_seed),
            str(run_spec.get("slug", "")),
            str(experiment_row.get("experiment_slug", "")),
            str(target.get("source_type", "")),
            str(target.get("source_slug", "")),
            str(target.get("intervention_source", "")),
            str(target.get("intervention_slug", "")),
            str(int(sample_index)),
        ]
    )
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _empty_sample_summary_accumulator() -> dict[str, list[object]]:
    return {
        "overall_mean_magnetization": [],
        "post_intervention_mean_magnetization": [],
        "unit_mean_magnetization": [],
        "time_mean_magnetization": [],
    }


def _append_sample_summary(
    accumulator: dict[str, list[object]],
    sample_summary: dict[str, object],
) -> None:
    for key in accumulator:
        accumulator[key].append(sample_summary[key])


def _finalize_sample_summaries(
    accumulator: dict[str, list[object]],
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(values, dtype=float)
        for key, values in accumulator.items()
    }


def _simulate_target(
    target: dict[str, object],
    run_spec: dict[str, object],
    overwrite: bool,
) -> dict[str, object]:
    """Execute the full simulation/writeout flow for one resolved target.

    `target` already identifies the experiment, the parameter source
    (truth/truth_xi_zero/fit), and which intervention panel to condition on.
    `run_spec` contributes simulation controls such as `num_samples`,
    `gibbs_sweeps`, and the base seed.
    """
    experiment_row = target["experiment_row"]
    experiment_root = Path(str(experiment_row["experiment_path"])).resolve()
    run_slug = str(run_spec["slug"])
    source_slug = str(target["source_slug"])
    intervention_source = str(target["intervention_source"])
    intervention_name = str(target["intervention_name"])
    intervention_slug = str(target["intervention_slug"])
    # Observed-experiment runs are evaluated as posterior predictive checks.
    # Saved-intervention runs are treated as counterfactuals and live under a
    # different output tree because they do not compare against realized x.
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

    # Load the realized experiment panel once, then resolve the intervention
    # source against that panel so saved interventions can be shape-checked.
    panel_context = load_experiment_panel_context(experiment_root)
    intervention_context = resolve_intervention_context(
        experiment_root,
        intervention_source=intervention_source,
        intervention_name=intervention_name,
        panel_context=panel_context,
    )
    # Choose the parameter source:
    # - truth: exact generating parameters for synthetic/hybrid experiments
    # - truth_xi_zero: truth with graph interactions disabled
    # - fit: estimated parameters loaded from a fit output directory
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
    # Posterior predictive draws must align with the experiment horizon because
    # all summaries and intervention panels are defined on that same T x N grid.
    if int(bundle.t_steps) != int(panel_context["T"]):
        raise ValueError(
            f"Posterior-predictive source '{target['source_name']}' has t_steps={bundle.t_steps},"
            f" but experiment '{experiment_row.get('experiment_name', experiment_root.name)}' has T={panel_context['T']}."
        )
    num_samples = int(run_spec["num_samples"])
    gibbs_sweeps = int(run_spec["gibbs_sweeps"])
    seed = int(run_spec["seed"])
    simulated_stats: list[dict[str, float | None]] = []
    counterfactual_sample_summaries = _empty_sample_summary_accumulator()
    observed_stats = None
    observed_sample_summary = None
    observed_draw_summaries = _empty_sample_summary_accumulator()
    # Only observed-experiment targets have realized outcomes to compare
    # against, so we precompute their reference statistics once up front.
    if intervention_source == "observed_experiment":
        observed_stats = compute_panel_statistics(
            panel_context["x"],
            z=intervention_context.z,
            x_0=panel_context["x_0"],
            s=int(intervention_context.s),
            field_matrix=bundle.field_matrix,
            gamma_matrix=bundle.gamma_matrix,
        )
        observed_sample_summary = compute_counterfactual_sample_summary(
            panel_context["x"],
            s=int(intervention_context.s),
        )
    for sample_index in range(num_samples):
        print(
            f"Simulating posterior-predictive sample {sample_index + 1} / {num_samples} for target '{target['source_name']}', intervention '{intervention_name}', and run '{run_spec['name']}'..."
        )
        sample_seed = _posterior_sample_seed(
            target=target,
            run_spec=run_spec,
            sample_index=sample_index,
        )
        sample_x = simulate_outcomes_for_bundle(
            bundle,
            x_0=panel_context["x_0"],
            z=intervention_context.z,
            gibbs_sweeps=gibbs_sweeps,
            seed=sample_seed,
        )
        if intervention_source == "observed_experiment":
            # Posterior predictive check: compare simulated panel-level
            # statistics and mean trajectories against the realized panel.
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
            sample_summary = compute_counterfactual_sample_summary(
                sample_x,
                s=int(intervention_context.s),
            )
            _append_sample_summary(observed_draw_summaries, sample_summary)
        else:
            # Counterfactual branch: there is no observed x under this
            # intervention, so we only keep across-draw summaries.
            sample_summary = compute_counterfactual_sample_summary(
                sample_x,
                s=int(intervention_context.s),
            )
            _append_sample_summary(counterfactual_sample_summaries, sample_summary)

    summary: dict[str, float | int | str] = {"s": int(intervention_context.s)}
    if intervention_source == "observed_experiment":
        # Write both:
        # 1. calibration-style panel statistics (`posterior_predictive_stats.csv`)
        # 2. draw-level mean summaries for overall/post/unit/time magnetization
        stat_rows, predictive_summary = summarize_predictive_statistics(
            observed_stats, simulated_stats
        )
        write_predictive_stats_tables(output_root, stat_rows)
        observed_sample_summaries = _finalize_sample_summaries(
            observed_draw_summaries
        )
        mean_rows, unit_rows, time_rows, mean_summary = (
            summarize_observed_mean_statistics(
                observed_sample_summary,
                observed_sample_summaries,
            )
        )
        write_observed_predictive_summary_tables(
            output_root,
            sample_summaries=observed_sample_summaries,
            mean_rows=mean_rows,
            unit_rows=unit_rows,
            time_rows=time_rows,
        )
        summary.update(predictive_summary)
        summary.update(mean_summary)
    else:
        # Counterfactual outputs only expose the simulated distribution because
        # there is no realized panel to score against.
        sample_summaries = _finalize_sample_summaries(
            counterfactual_sample_summaries
        )
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
    # Persist enough metadata for downstream reporting to rebuild the unified
    # manifest without having to rerun the simulation.
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
        "seed_strategy": "blake2b(base_seed, run_slug, target identity, sample_index)",
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
    """Resolve one CLI/API request into a single simulation target and run it.

    The pathway is:
    generation manifest -> fit manifest lookup -> target-pairs expansion ->
    unique target selection -> run-spec lookup -> simulation/writeout.
    """
    # The generation manifest anchors experiment paths; the fit manifest is
    # only needed for `source_type="fit"` targets.
    generation_lookup = index_generation_rows(generation_manifest_path)
    fit_lookup = resolve_fit_lookup(fit_manifest_path)
    # `target_pairs` is the explicit contract that says which source/intervention
    # combinations are valid to run for each experiment.
    targets = resolve_target_pairs(target_pairs_path, generation_lookup, fit_lookup)
    target = select_target(
        targets,
        experiment_name=experiment_name,
        source_type=source_type,
        variant_name=variant_name,
        intervention_source=intervention_source,
        intervention_name=intervention_name,
    )
    # `spec_path` can define multiple named posterior-predictive runs; this
    # worker executes exactly one of them.
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
