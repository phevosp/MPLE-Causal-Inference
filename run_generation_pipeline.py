from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse

from data.synthetic_data_generation import (
    derive_pre_intervention_steps,
    materialize_generation_experiment,
)
from pipeline_specs import expand_named_entries, slugify, write_csv_manifest


def _maybe_load_fixed_graph_shape(spec: dict[str, Any]) -> int | None:
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
) -> tuple[int | None, int | None, int | None]:
    intervention = spec.get("intervention", {})
    if intervention.get("source", "generated") != "fixed_artifact":
        return None, None, None
    artifact = intervention.get("artifact", {})
    panel_path = Path(str(artifact.get("panel_path", "")))
    z0_path = Path(str(artifact.get("z0_path", "")))
    if not panel_path.exists() or not z0_path.exists():
        raise FileNotFoundError(
            "Fixed intervention artifacts require both panel_path and z0_path."
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
    return int(z.shape[1]), int(z.shape[0]), derive_pre_intervention_steps(z)


def resolve_dimensions(spec: dict[str, Any]) -> dict[str, int]:
    dims = dict(spec.get("dimensions", {}) or {})
    graph_n = _maybe_load_fixed_graph_shape(spec)
    z_n, z_t, z_s = _maybe_load_fixed_intervention_shape(spec)
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

    intervention_source = spec.get("intervention", {}).get("source", "generated")
    if intervention_source == "fixed_artifact":
        s_value = int(z_s)
    else:
        if dims.get("s") is None:
            raise ValueError(
                f"Experiment '{spec['name']}' must define s when intervention source is generated."
            )
        s_value = int(dims["s"])
    return {"N": n_value, "T": t_value, "s": s_value}


def translate_generation_spec(spec: dict[str, Any]):
    dims = resolve_dimensions(spec)
    truth = dict(spec.get("truth", {}) or {})
    scalars = dict(truth.get("scalars", {}) or {})
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
    if intervention_source == "generated":
        intervention_mode = "generated_z"
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
                "s": dims["s"],
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
        "s": dims["s"],
        "has_truth": True,
        "latent_rank": int(spec.get("truth", {}).get("latent_rank", 0)),
    }


def run_generation(spec_path: str | Path, overwrite: bool = False) -> Path:
    experiments = expand_named_entries(spec_path, "experiments")
    if not experiments:
        raise ValueError(f"No experiments found in generation spec {spec_path}.")

    experiment_root = Path(str(experiments[0]["experiment_root"]))
    manifest_path = Path(str(experiments[0]["manifest_path"]))
    manifest_rows: list[dict[str, object]] = []

    for experiment_spec in experiments:
        config, dims = translate_generation_spec(experiment_spec)
        data_folder = experiment_root / slugify(experiment_spec["name"])
        if data_folder.exists():
            if overwrite:
                shutil.rmtree(data_folder)
            else:
                raise FileExistsError(
                    f"{data_folder} already exists. Re-run with --overwrite to rebuild it."
                )
        materialize_generation_experiment(
            config=config,
            data_folder=data_folder,
            descriptor=experiment_spec["name"],
            config_label=str(Path(spec_path).resolve()),
            extra_metadata={
                "experiment_name": experiment_spec["name"],
                "generation_spec_path": str(Path(spec_path).resolve()),
                "intervention_source": experiment_spec.get("intervention", {}).get(
                    "source", "generated"
                ),
                "graph_source": experiment_spec.get("graph", {}).get(
                    "source", "generated"
                ),
                "experiment_root": str(experiment_root),
            },
            config_filename="generation_realized_config.yaml",
        )
        manifest_rows.append(
            manifest_row_for_experiment(experiment_spec, data_folder, dims)
        )

    write_csv_manifest(manifest_path, manifest_rows)
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
    args = parser.parse_args()

    manifest_path = run_generation(args.spec_path, overwrite=args.overwrite)
    print(f"Generation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
