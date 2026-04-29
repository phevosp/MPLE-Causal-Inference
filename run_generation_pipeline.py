"""Materialize synthetic experiment artifacts from a generation_spec.yaml.

Supports three complementary workflows:

- plan generation requests into ``generation_requests.csv``
- execute one planned generation request by experiment slug
- refresh ``generation_manifest.csv`` from completed experiment outputs

The sequential ``run_generation(...)`` entry point now reuses the same
request-planning, single-request execution, and manifest-refresh helpers used by
the shell/SLURM wrappers.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse

from data.synthetic_data_generation import (
    materialize_generation_experiment,
)
from pipeline_specs import (
    expand_named_entries,
    read_csv_manifest,
    slugify,
    write_csv_manifest,
)


GENERATION_REQUESTS_NAME = "generation_requests.csv"


def _read_yaml_mapping(path: str | Path) -> dict[str, object]:
    loaded = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping data in {path}.")
    return loaded


def _expand_generation_experiments(spec_path: str | Path) -> list[dict[str, Any]]:
    experiments = expand_named_entries(spec_path, "experiments")
    if not experiments:
        raise ValueError(f"No experiments found in generation spec {spec_path}.")
    return experiments


def generation_manifest_path_for_spec(spec_path: str | Path) -> Path:
    experiments = _expand_generation_experiments(spec_path)
    return Path(str(experiments[0]["manifest_path"]))


def generation_requests_path_for_spec(spec_path: str | Path) -> Path:
    return generation_manifest_path_for_spec(spec_path).with_name(
        GENERATION_REQUESTS_NAME
    )


def _generation_request_row(
    experiment_spec: dict[str, Any],
    spec_path: str | Path,
) -> dict[str, object]:
    experiment_root = Path(str(experiment_spec["experiment_root"]))
    experiment_slug = slugify(str(experiment_spec["name"]))
    experiment_path = experiment_root / experiment_slug
    return {
        "generation_spec_path": str(Path(spec_path).resolve()),
        "experiment_name": str(experiment_spec["name"]),
        "experiment_slug": experiment_slug,
        "experiment_path": str(experiment_path.resolve()),
    }


def write_generation_requests(spec_path: str | Path) -> Path:
    experiments = _expand_generation_experiments(spec_path)
    request_path = generation_requests_path_for_spec(spec_path)
    request_rows = [
        _generation_request_row(experiment_spec, spec_path)
        for experiment_spec in experiments
    ]
    write_csv_manifest(request_path, request_rows)
    return request_path


def _select_generation_experiment(
    spec_path: str | Path,
    experiment_slug: str,
) -> dict[str, Any]:
    matches = [
        experiment_spec
        for experiment_spec in _expand_generation_experiments(spec_path)
        if slugify(str(experiment_spec["name"])) == experiment_slug
    ]
    if not matches:
        raise ValueError(
            f"No generation experiment with slug '{experiment_slug}' found in {spec_path}."
        )
    if len(matches) != 1:
        raise ValueError(
            f"Generation experiment slug '{experiment_slug}' is not unique in {spec_path}."
        )
    return matches[0]


def _maybe_load_fixed_graph_shape(spec: dict[str, Any]) -> int | None:
    """Load shape of fixed graph, if source is fixed artifact"""
    graph = spec.get("graph", {})
    if graph.get("source", "generated") != "fixed_artifact":
        return None
    graph_path = Path(str(graph.get("artifact", {}).get("gamma_path", "")))
    if not graph_path.exists():
        raise FileNotFoundError(f"Fixed graph artifact not found: {graph_path}")
    if graph_path.suffix == ".npz":
        matrix = sparse.load_npz(graph_path)
    else:
        matrix = np.asarray(np.load(graph_path), dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Fixed graph must be square: {graph_path}")
    return int(matrix.shape[0])


def _maybe_load_fixed_intervention_shape(
    spec: dict[str, Any],
) -> tuple[int | None, int | None]:
    """Load shape of fixed intervention panel, if source is fixed artifact"""
    intervention = spec.get("intervention", {})
    if intervention.get("source", "generated") != "fixed_artifact":
        return None, None
    artifact = intervention.get("artifact", {})
    panel_path = Path(str(artifact.get("panel_path", "")))
    z0_path = Path(str(artifact.get("z0_path", "")))
    if not panel_path.exists() or not z0_path.exists():
        missing = [
            f"{label}={path}"
            for label, path in [("panel_path", panel_path), ("z0_path", z0_path)]
            if not path.exists()
        ]
        raise FileNotFoundError(
            "Fixed intervention artifacts require both panel_path and z0_path; "
            f"missing {', '.join(missing)}."
        )
    with np.load(panel_path, allow_pickle=False) as data:
        if "z" not in data:
            raise KeyError(f"Fixed intervention panel missing 'z': {panel_path}")
        z = np.asarray(data["z"], dtype=float)
    z_0 = np.asarray(np.load(z0_path), dtype=float)
    if z.ndim != 2:
        raise ValueError(f"Fixed intervention z must be 2D: {panel_path}")
    if z_0.shape != (z.shape[1],):
        raise ValueError(
            f"Fixed z_0 shape {z_0.shape} does not match panel width {z.shape[1]}."
        )
    return int(z.shape[1]), int(z.shape[0])


def resolve_dimensions(spec: dict[str, Any]) -> dict[str, int]:
    """Resolve dimensions N and T for an experiment spec."""
    dims = dict(spec.get("dimensions", {}) or {})
    graph_n = _maybe_load_fixed_graph_shape(spec)
    z_n, z_t = _maybe_load_fixed_intervention_shape(spec)
    fixed_n_candidates = [value for value in [graph_n, z_n] if value is not None]
    if fixed_n_candidates and len(set(int(value) for value in fixed_n_candidates)) != 1:
        raise ValueError(
            f"Experiment '{spec['name']}' has inconsistent fixed N sources."
        )

    if fixed_n_candidates:
        # Fixed artifacts determine N; ignore inherited/default dimensions.N mismatches.
        n_value = int(fixed_n_candidates[0])
    elif dims.get("N") is not None:
        n_value = int(dims["N"])
    else:
        raise ValueError(
            f"Experiment '{spec['name']}' must define N or provide a fixed source that implies it."
        )

    if z_t is not None:
        # Fixed intervention panels determine T; ignore inherited/default dimensions.T mismatches.
        t_value = int(z_t)
    elif dims.get("T") is not None:
        t_value = int(dims["T"])
    else:
        raise ValueError(
            f"Experiment '{spec['name']}' must define T or provide a fixed intervention source."
        )

    return {"N": n_value, "T": t_value}


def translate_generation_spec(spec: dict[str, Any]):
    """Translate an experiment spec into a config dict for generation."""
    dims = resolve_dimensions(spec)
    truth = dict(spec.get("truth", {}) or {})
    scalars = dict(truth.get("scalars", {}) or {})
    field_mode = str(truth.get("field_mode", "random_low_rank"))
    field_params = dict(truth.get("field_params", {}) or {})
    generation = dict(spec.get("generation", {}) or {})
    x0 = dict(spec.get("x0", {}) or {})
    graph = dict(spec.get("graph", {}) or {})
    intervention = dict(spec.get("intervention", {}) or {})

    graph_source = str(graph.get("source", "generated"))
    graph_artifact = dict(graph.get("artifact", {}) or {})
    if graph_source == "generated":
        gamma_matrix_generator = str(graph["generator"])
        gamma_matrix_params = dict(graph.get("params", {}) or {})
    elif graph_source == "fixed_artifact":
        gamma_matrix_generator = "fixed_artifact"
        gamma_matrix_params = dict(graph.get("params", {}) or {})
    else:
        raise ValueError(f"Unknown graph source '{graph_source}'.")

    intervention_source = str(intervention.get("source", "generated"))
    intervention_artifact = dict(intervention.get("artifact", {}) or {})
    intervention_generator = str(
        intervention.get("generator", "low_rank_probability")
    ).strip()
    intervention_params = dict(intervention.get("params", {}) or {})
    if intervention_source == "generated":
        if intervention_generator != "low_rank_probability":
            raise ValueError(
                "Generated interventions only support intervention.generator='low_rank_probability'."
            )
        intervention_mode = "low_rank_probability"
    elif intervention_source == "fixed_artifact":
        intervention_mode = "fixed_z"
    else:
        raise ValueError(f"Unknown intervention source '{intervention_source}'.")

    latent_rank = int(truth.get("latent_rank", 0))
    if latent_rank < 0:
        raise ValueError("truth.latent_rank must be nonnegative.")

    config = OmegaConf.create(
        {
            "global_params": {
                "N": dims["N"],
                "T": dims["T"],
                "B": float(truth["B"]),
                "gamma_matrix_generator": gamma_matrix_generator,
                "fixed_gamma_source": {
                    "gamma_path": graph_artifact.get("gamma_path"),
                    "node_index_path": graph_artifact.get("node_index_path"),
                    "artifact_dir": graph_artifact.get("artifact_dir"),
                    "network_name": graph_artifact.get("network_name"),
                    "trim_scope": graph_artifact.get("trim_scope"),
                },
                "x_0_generator": str(x0["generator"]),
                "latent_rank": latent_rank,
                "field_mode": field_mode,
                "field_params": field_params,
                "gamma_matrix_params": gamma_matrix_params,
                "x_0_params": dict(x0.get("params", {}) or {}),
            },
            "estimation_params": {
                "beta": float(scalars["beta"]),
                "xi": float(scalars["xi"]),
                "eta": float(scalars["eta"]),
                "zeta": float(scalars["zeta"]),
                "psi": float(scalars["psi"]),
            },
            "generation_params": {
                "seed": int(generation["seed"]),
                "gibbs_sweeps": int(generation["gibbs_sweeps"]),
                "intervention_mode": intervention_mode,
                "intervention_generator": intervention_generator,
                "intervention_params": intervention_params,
                "fixed_z_source": {
                    "panel_path": intervention_artifact.get("panel_path"),
                    "z0_path": intervention_artifact.get("z0_path"),
                    "artifact_dir": intervention_artifact.get("artifact_dir"),
                    "shared_panel_dir": intervention_artifact.get("shared_panel_dir"),
                    "outcome_code": intervention_artifact.get("outcome_code"),
                    "intervention_code": intervention_artifact.get("intervention_code"),
                    "lag_code": intervention_artifact.get("lag_code"),
                    "trim_scope": intervention_artifact.get("trim_scope"),
                },
            },
        }
    )

    return config, dims


def manifest_row_for_experiment(
    spec: dict[str, Any],
    data_folder: Path,
    dims: dict[str, int],
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment_name": spec["name"],
        "experiment_slug": slugify(spec["name"]),
        "descriptor": spec["name"],
        "experiment_path": str(data_folder.resolve()),
        "intervention_source": str(
            spec.get("intervention", {}).get("source", "generated")
        ),
        "graph_source": str(spec.get("graph", {}).get("source", "generated")),
        "N": dims["N"],
        "T": dims["T"],
        "has_truth": True,
        "field_mode": str(metadata.get("field_mode", "random_low_rank")),
        "latent_rank": int(
            metadata.get(
                "latent_rank", spec.get("truth", {}).get("latent_rank", 0)
            )
        ),
    }


def run_generation_request(
    spec_path: str | Path,
    experiment_slug: str,
    overwrite: bool = False,
) -> dict[str, object]:
    experiment_spec = _select_generation_experiment(spec_path, experiment_slug)
    config, dims = translate_generation_spec(experiment_spec)
    experiment_root = Path(str(experiment_spec["experiment_root"]))
    data_folder = experiment_root / slugify(str(experiment_spec["name"]))
    if data_folder.exists():
        if overwrite:
            shutil.rmtree(data_folder)
        else:
            raise FileExistsError(
                f"{data_folder} already exists. Re-run with --overwrite to rebuild it."
            )
    metadata = materialize_generation_experiment(
        config=config,
        data_folder=data_folder,
        descriptor=str(experiment_spec["name"]),
        config_label=str(Path(spec_path).resolve()),
        extra_metadata={
            "experiment_name": experiment_spec["name"],
            "generation_spec_path": str(Path(spec_path).resolve()),
            "intervention_source": experiment_spec.get("intervention", {}).get(
                "source", "generated"
            ),
            "graph_source": experiment_spec.get("graph", {}).get("source", "generated"),
            "experiment_root": str(experiment_root),
        },
        config_filename="generation_realized_config.yaml",
    )
    return manifest_row_for_experiment(experiment_spec, data_folder, dims, metadata)


def _manifest_row_from_completed_experiment(
    request_row: dict[str, str],
) -> dict[str, object]:
    experiment_root = Path(request_row["experiment_path"]).resolve()
    metadata_path = experiment_root / "experiment_metadata.yaml"
    panel_path = experiment_root / "panel_data.npz"
    field_artifacts_path = experiment_root / "field_artifacts.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing experiment metadata: {metadata_path}")
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing panel artifact: {panel_path}")
    if not field_artifacts_path.exists():
        raise FileNotFoundError(f"Missing field artifacts: {field_artifacts_path}")

    metadata = _read_yaml_mapping(metadata_path)
    with np.load(panel_path, allow_pickle=False) as panel_data:
        x = np.asarray(panel_data["x"], dtype=float)
        z = np.asarray(panel_data["z"], dtype=float)
    with np.load(field_artifacts_path, allow_pickle=False) as field_data:
        latent_rank = int(np.asarray(field_data["latent_rank"]).item())

    if x.ndim != 2 or z.shape != x.shape:
        raise ValueError(
            f"Completed experiment panel must contain matching 2D x/z arrays: {panel_path}"
        )

    experiment_name = str(
        metadata.get("experiment_name", request_row.get("experiment_name", ""))
    )
    return {
        "experiment_name": experiment_name,
        "experiment_slug": str(
            metadata.get("slug", request_row.get("experiment_slug", slugify(experiment_name)))
        ),
        "descriptor": str(metadata.get("descriptor", experiment_name)),
        "experiment_path": str(experiment_root),
        "intervention_source": str(metadata.get("intervention_source", "")),
        "graph_source": str(metadata.get("graph_source", "")),
        "N": int(x.shape[1]),
        "T": int(x.shape[0]),
        "has_truth": bool(metadata.get("has_truth", True)),
        "field_mode": str(metadata.get("field_mode", "random_low_rank")),
        "latent_rank": latent_rank,
    }


def refresh_generation_manifest(spec_path: str | Path) -> Path:
    request_path = generation_requests_path_for_spec(spec_path)
    if not request_path.exists():
        write_generation_requests(spec_path)
    request_rows = read_csv_manifest(request_path)
    manifest_rows = [
        _manifest_row_from_completed_experiment(request_row)
        for request_row in request_rows
    ]
    manifest_path = generation_manifest_path_for_spec(spec_path)
    write_csv_manifest(manifest_path, manifest_rows)
    return manifest_path


def run_generation(spec_path: str | Path, overwrite: bool = False) -> Path:
    request_path = write_generation_requests(spec_path)
    request_rows = read_csv_manifest(request_path)
    print(f"Loaded {len(request_rows)} generation experiment(s) from {Path(spec_path)}.")
    for request_row in request_rows:
        experiment_name = request_row.get("experiment_name", request_row["experiment_slug"])
        print(f"Preparing experiment '{experiment_name}'...")
        run_generation_request(
            spec_path,
            request_row["experiment_slug"],
            overwrite=overwrite,
        )
    manifest_path = refresh_generation_manifest(spec_path)
    print(f"Wrote generation manifest to {manifest_path}.")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize synthetic and hybrid generation experiments from a YAML spec."
    )
    parser.add_argument(
        "--spec_path",
        type=str,
        default="data/configs/generation_spec.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--write_requests",
        action="store_true",
        help="Write generation_requests.csv for the configured generation spec.",
    )
    parser.add_argument(
        "--run_request",
        action="store_true",
        help="Run one planned generation request selected by --experiment_slug.",
    )
    parser.add_argument(
        "--refresh_manifest",
        action="store_true",
        help="Refresh generation_manifest.csv from completed experiment outputs.",
    )
    parser.add_argument(
        "--experiment_slug",
        type=str,
        default="",
        help="Slug of the experiment to materialize when --run_request is set.",
    )
    args = parser.parse_args()

    if args.write_requests:
        request_path = write_generation_requests(args.spec_path)
        print(f"Generation requests: {request_path}")
        return

    if args.run_request:
        if not args.experiment_slug.strip():
            raise ValueError("--experiment_slug is required when --run_request is set.")
        row = run_generation_request(
            args.spec_path,
            args.experiment_slug.strip(),
            overwrite=args.overwrite,
        )
        print(f"Generated experiment: {row['experiment_path']}")
        return

    if args.refresh_manifest:
        manifest_path = refresh_generation_manifest(args.spec_path)
        print(f"Generation manifest: {manifest_path}")
        return

    manifest_path = run_generation(args.spec_path, overwrite=args.overwrite)
    print(f"Generation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
