from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from utils.t6_intervention_utils import (
    INTERVENTION_LIBRARY_ROOT_NAME,
    build_full_on_intervention,
    build_single_unit_on_intervention,
    save_intervention_artifact,
)
from utils.t5_experiment_context import load_experiment_panel_context
from utils.t0_csv_utils import read_csv_rows, write_csv_rows
from utils.t6_pipeline_spec_utils import expand_named_entries


def _generation_manifest_rows(
    generation_manifest_path: str | Path,
) -> list[dict[str, str]]:
    rows = read_csv_rows(generation_manifest_path)
    if not rows:
        raise ValueError(
            f"No rows found in generation manifest {generation_manifest_path}."
        )
    seen_experiment_names: set[str] = set()
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        experiment_name = str(row.get("experiment_name", "")).strip()
        if not experiment_name:
            raise ValueError(
                f"Generation manifest {generation_manifest_path} contains a row without experiment_name."
            )
        experiment_path = str(row.get("experiment_path", "")).strip()
        if not experiment_path:
            raise ValueError(
                f"Generation manifest {generation_manifest_path} contains a row without experiment_path."
            )
        if experiment_name in seen_experiment_names:
            raise ValueError(
                f"Generation manifest {generation_manifest_path} contains duplicate experiment_name '{experiment_name}'."
            )
        seen_experiment_names.add(experiment_name)
        normalized_rows.append(row)
    return normalized_rows


def _validate_intervention_entries(
    entries: list[dict[str, object]],
    spec_path: str | Path,
) -> None:
    stale_experiment_targets = [
        str(entry["name"])
        for entry in entries
        if str(entry.get("experiment_name", "")).strip()
    ]
    if stale_experiment_targets:
        raise ValueError(
            f"Intervention-library spec {spec_path} should not define experiment_name. "
            "Interventions are now materialized for every experiment in the generation manifest."
        )


def _materialize_saved_intervention(
    entry: dict[str, object],
    experiment_row: dict[str, str],
    *,
    overwrite: bool,
) -> dict[str, object]:
    experiment_name = str(experiment_row["experiment_name"]).strip()
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
    _validate_intervention_entries(entries, spec_path)
    generation_rows = _generation_manifest_rows(generation_manifest_path)
    manifest_rows = [
        _materialize_saved_intervention(
            entry,
            experiment_row,
            overwrite=overwrite,
        )
        for experiment_row in generation_rows
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
    write_csv_rows(manifest_path, manifest_rows)
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
