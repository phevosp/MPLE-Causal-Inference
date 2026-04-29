from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from intervention_utils import (
    INTERVENTION_LIBRARY_ROOT_NAME,
    build_full_on_intervention,
    build_single_unit_on_intervention,
    save_intervention_artifact,
)
from loading_utils import load_experiment_panel_context
from pipeline_specs import expand_named_entries, read_csv_manifest, write_csv_manifest


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


def _materialize_saved_intervention(
    entry: dict[str, object],
    generation_lookup: dict[str, dict[str, str]],
    *,
    overwrite: bool,
) -> dict[str, object]:
    experiment_name = str(entry["experiment_name"])
    experiment_row = generation_lookup.get(experiment_name)
    if experiment_row is None:
        raise ValueError(
            f"Intervention spec references unknown experiment '{experiment_name}'."
        )
    experiment_root = Path(str(experiment_row["experiment_path"])).resolve()
    panel_context = load_experiment_panel_context(experiment_root)
    source_kind = str(entry["source_kind"]).strip().lower()

    if source_kind == "observed_experiment":
        z = panel_context["z"]
        z_0 = panel_context["z_0"]
        s_value = int(panel_context["s"])
        e_value = int(panel_context["e"])
        extra_metadata = {}
    elif source_kind == "full_on":
        activation_scope = str(entry["activation_scope"]).strip().lower()
        z, z_0, s_value, e_value = build_full_on_intervention(
            int(panel_context["N"]),
            int(panel_context["T"]),
            int(panel_context["s"]),
            activation_scope=activation_scope,
        )
        extra_metadata = {"activation_scope": activation_scope}
    elif source_kind == "single_unit_on":
        activation_scope = str(entry["activation_scope"]).strip().lower()
        unit_index = int(entry["unit_index"])
        start_step = entry.get("start_step")
        z, z_0, s_value, e_value = build_single_unit_on_intervention(
            int(panel_context["N"]),
            int(panel_context["T"]),
            int(panel_context["s"]),
            unit_index=unit_index,
            activation_scope=activation_scope,
            start_step=None if start_step in (None, "") else int(start_step),
        )
        extra_metadata = {
            "activation_scope": activation_scope,
            "unit_index": unit_index,
        }
        if start_step not in (None, ""):
            extra_metadata["start_step"] = int(start_step)
    else:
        raise ValueError(f"Unsupported source_kind '{source_kind}'.")

    output_root = experiment_root / INTERVENTION_LIBRARY_ROOT_NAME / str(entry["slug"])
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"{output_root} already exists. Re-run with --overwrite to rebuild it."
            )
    save_intervention_artifact(
        output_root,
        intervention_name=str(entry["name"]),
        experiment_name=experiment_name,
        z=z,
        z_0=z_0,
        s=int(s_value),
        e=int(e_value),
        source_kind=source_kind,
        extra_metadata=extra_metadata,
    )
    return {
        "experiment_name": experiment_name,
        "experiment_path": str(experiment_root),
        "intervention_name": str(entry["name"]),
        "intervention_slug": str(entry["slug"]),
        "source_kind": source_kind,
        "N": int(panel_context["N"]),
        "T": int(panel_context["T"]),
        "s": int(s_value),
        "e": int(e_value),
        "output_path": str(output_root),
        "activation_scope": extra_metadata.get("activation_scope", ""),
        "unit_index": extra_metadata.get("unit_index", ""),
        "start_step": extra_metadata.get("start_step", ""),
    }


def run_intervention_library(
    generation_manifest_path: str | Path,
    spec_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    entries = expand_named_entries(spec_path, "interventions")
    if not entries:
        raise ValueError(f"No interventions found in intervention-library spec {spec_path}.")
    generation_lookup = _index_generation_rows(generation_manifest_path)
    manifest_rows = [
        _materialize_saved_intervention(
            entry,
            generation_lookup,
            overwrite=overwrite,
        )
        for entry in entries
    ]
    manifest_path = Path(
        str(
            entries[0].get(
                "manifest_path",
                Path(generation_manifest_path).resolve().parent
                / "intervention_library_manifest.csv",
            )
        )
    )
    write_csv_manifest(manifest_path, manifest_rows)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize reusable intervention panels for generated experiments."
    )
    parser.add_argument("--generation_manifest_path", required=True, type=str)
    parser.add_argument(
        "--spec_path",
        type=str,
        default="data/configs/intervention_library_spec.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = run_intervention_library(
        args.generation_manifest_path,
        args.spec_path,
        overwrite=args.overwrite,
    )
    print(f"Intervention library manifest: {manifest_path}")


if __name__ == "__main__":
    main()
