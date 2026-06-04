"""Intervention construction and intervention-artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from utils.io_utils import io_path


INTERVENTION_LIBRARY_ROOT_NAME = "intervention_library"
COUNTERFACTUAL_ROOT_NAME = "counterfactual"
COUNTERFACTUAL_MANIFEST_NAME = "counterfactual_manifest.csv"


@dataclass(frozen=True)
class InterventionContext:
    source_kind: str
    intervention_name: str
    intervention_slug: str
    z: np.ndarray
    z_0: np.ndarray
    s: int
    e: int
    metadata: dict[str, object]


def derive_pre_intervention_steps(z: np.ndarray) -> int:
    treated_rows = np.any(np.asarray(z) == 1, axis=1)
    return int(np.argmax(treated_rows)) if treated_rows.any() else int(z.shape[0])


def derive_post_intervention_steps(z: np.ndarray) -> int:
    # e = first time step AFTER the last unit transitions from untreated (-1) to treated (+1)
    # Find when each unit first gets treated (first occurrence of z=1 for that unit)
    z_array = np.asarray(z, dtype=float)
    T, N = z_array.shape

    # Vectorized: find first treatment time for each unit
    treatment_mask = z_array == 1.0
    # argmax gives index of first True; if no True, gives 0 (which we handle below)
    first_treatment_per_unit = np.argmax(treatment_mask, axis=0)
    # Mark units never treated with T
    never_treated = ~np.any(treatment_mask, axis=0)
    first_treatment_per_unit[never_treated] = T

    # If no units are ever treated, e=0
    if np.all(first_treatment_per_unit >= T):
        return 0

    # e = time of last unit's first treatment + 1
    last_unit_first_treatment = int(np.max(first_treatment_per_unit[first_treatment_per_unit < T]))
    return last_unit_first_treatment + 1


def _validate_intervention_panel(z: np.ndarray, z_0: np.ndarray) -> None:
    if z.ndim != 2:
        raise ValueError("Intervention panel z must be 2D.")
    if z_0.shape != (z.shape[1],):
        raise ValueError(
            f"Intervention z_0 shape {z_0.shape} does not match panel width {z.shape[1]}."
        )
    if not np.all(np.isin(z, (-1.0, 1.0))):
        raise ValueError("Intervention panel z must use -1/+1 coding only.")
    if not np.all(np.isin(z_0, (-1.0, 0.0, 1.0))):
        raise ValueError("Intervention z_0 must use legacy 0 or -1/+1 coding only.")


def save_intervention_artifact(
    output_root: str | Path,
    *,
    intervention_name: str,
    experiment_name: str,
    z: np.ndarray,
    z_0: np.ndarray,
    s: int,
    e: int | None = None,
    source_kind: str,
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    from pipeline_specs import slugify

    artifact_root = Path(output_root)
    z = np.asarray(z, dtype=float)
    z_0 = np.asarray(z_0, dtype=float)
    _validate_intervention_panel(z, z_0)
    artifact_root.mkdir(parents=True, exist_ok=False)
    np.savez(artifact_root / "intervention_panel.npz", z=z)
    np.save(artifact_root / "z_0.npy", z_0)
    if e is None:
        e = derive_post_intervention_steps(z)
    metadata = {
        "intervention_name": intervention_name,
        "intervention_slug": slugify(intervention_name),
        "experiment_name": experiment_name,
        "N": int(z.shape[1]),
        "T": int(z.shape[0]),
        "s": int(s),
        "e": int(e),
        "source_kind": source_kind,
        **dict(extra_metadata or {}),
    }
    with open(
        io_path(artifact_root / "intervention_metadata.yaml"), "w", encoding="utf-8"
    ) as handle:
        OmegaConf.save(OmegaConf.create(metadata), handle)
    return artifact_root


def build_full_on_intervention(
    n_nodes: int,
    t_steps: int,
    s: int,
    *,
    activation_scope: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    z = -np.ones((int(t_steps), int(n_nodes)), dtype=float)
    z_0 = -np.ones(int(n_nodes), dtype=float)
    scope = str(activation_scope)
    if scope == "all_time":
        z[:, :] = 1.0
        z_0[:] = 1.0
    elif scope == "no_time":
        pass
    elif scope == "from_s":
        z[int(s) :, :] = 1.0
    else:
        raise ValueError(f"Unsupported full_on activation_scope '{activation_scope}'.")
    return z, z_0, derive_pre_intervention_steps(z), derive_post_intervention_steps(z)


def build_single_unit_on_intervention(
    n_nodes: int,
    t_steps: int,
    s: int,
    *,
    unit_index: int,
    activation_scope: str,
    start_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if int(unit_index) < 0 or int(unit_index) >= int(n_nodes):
        raise ValueError(
            f"unit_index={unit_index} is out of bounds for N={n_nodes}."
        )
    z = -np.ones((int(t_steps), int(n_nodes)), dtype=float)
    z_0 = -np.ones(int(n_nodes), dtype=float)
    scope = str(activation_scope)
    if scope == "all_time":
        z[:, int(unit_index)] = 1.0
        z_0[int(unit_index)] = 1.0
    elif scope == "no_time":
        pass
    elif scope == "from_s":
        z[int(s) :, int(unit_index)] = 1.0
    elif scope == "from_step":
        if start_step is None:
            raise ValueError("start_step is required when activation_scope='from_step'.")
        if int(start_step) < 0 or int(start_step) > int(t_steps):
            raise ValueError(
                f"start_step={start_step} must lie in [0, {t_steps}]."
            )
        z[int(start_step) :, int(unit_index)] = 1.0
    else:
        raise ValueError(
            f"Unsupported single_unit_on activation_scope '{activation_scope}'."
        )
    return z, z_0, derive_pre_intervention_steps(z), derive_post_intervention_steps(z)


def load_saved_intervention_context(
    experiment_root: str | Path,
    intervention_name: str,
) -> InterventionContext:
    from pipeline_specs import slugify

    experiment_path = Path(experiment_root)
    intervention_slug = slugify(intervention_name)
    artifact_root = experiment_path / INTERVENTION_LIBRARY_ROOT_NAME / intervention_slug
    panel_path = artifact_root / "intervention_panel.npz"
    z0_path = artifact_root / "z_0.npy"
    metadata_path = artifact_root / "intervention_metadata.yaml"
    if not panel_path.exists() or not z0_path.exists():
        raise FileNotFoundError(
            f"Saved intervention artifact '{intervention_name}' not found under {artifact_root}."
        )
    with np.load(panel_path, allow_pickle=False) as data:
        if "z" not in data:
            raise KeyError(f"Intervention artifact {panel_path} does not contain 'z'.")
        z = np.asarray(data["z"], dtype=float)
    z_0 = np.asarray(np.load(z0_path), dtype=float)
    _validate_intervention_panel(z, z_0)
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        with open(io_path(metadata_path), "r", encoding="utf-8") as handle:
            loaded = OmegaConf.to_container(OmegaConf.load(handle), resolve=True)
        if isinstance(loaded, dict):
            metadata = loaded
    return InterventionContext(
        source_kind="saved_intervention",
        intervention_name=str(metadata.get("intervention_name", intervention_name)),
        intervention_slug=str(metadata.get("intervention_slug", intervention_slug)),
        z=z,
        z_0=z_0,
        s=int(metadata.get("s", derive_pre_intervention_steps(z))),
        e=int(metadata.get("e", derive_post_intervention_steps(z))),
        metadata=metadata,
    )


def resolve_intervention_context(
    experiment_root: str | Path,
    *,
    intervention_source: str,
    intervention_name: str | None = None,
    panel_context: dict[str, object] | None = None,
) -> InterventionContext:
    source = str(intervention_source).strip().lower()
    if source == "observed_experiment":
        from utils.loading_utils import load_experiment_panel_context

        resolved_panel_context = (
            panel_context
            if panel_context is not None
            else load_experiment_panel_context(experiment_root)
        )
        return InterventionContext(
            source_kind="observed_experiment",
            intervention_name="observed_experiment",
            intervention_slug="observed_experiment",
            z=np.asarray(resolved_panel_context["z"], dtype=float),
            z_0=np.asarray(resolved_panel_context["z_0"], dtype=float),
            s=int(resolved_panel_context["s"]),
            e=int(resolved_panel_context.get("e", derive_post_intervention_steps(resolved_panel_context["z"]))),
            metadata={"source_kind": "observed_experiment"},
        )
    if source == "saved_intervention":
        if not intervention_name or not str(intervention_name).strip():
            raise ValueError(
                "saved_intervention targets must provide intervention_name."
            )
        context = load_saved_intervention_context(experiment_root, intervention_name)
        if panel_context is not None and context.z.shape != (
            int(panel_context["T"]),
            int(panel_context["N"]),
        ):
            raise ValueError(
                f"Saved intervention '{intervention_name}' has shape {context.z.shape},"
                f" expected {(int(panel_context['T']), int(panel_context['N']))}."
            )
        return context
    raise ValueError(f"Unsupported intervention_source '{intervention_source}'.")

