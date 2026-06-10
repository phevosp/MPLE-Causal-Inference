"""Plot posterior-predictive and intervention time-mean summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.t0_csv_utils import read_csv_rows
from utils.t0_orcd_path_remap import resolve_orcd_local_path
from utils.t5_experiment_context import load_experiment_panel_context


def _time_summary_rows(csv_path: str | Path) -> list[dict[str, str]]:
    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No time-summary rows found in {csv_path}.")
    return rows


def _time_axis(rows: list[dict[str, str]]) -> np.ndarray:
    axis: list[int] = []
    for index, row in enumerate(rows):
        value = row.get("time_index", "")
        axis.append(index if value in (None, "") else int(value))
    return np.asarray(axis, dtype=int)


def _float_series(
    rows: list[dict[str, str]],
    column: str,
    *,
    csv_path: str | Path,
) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(column, "")
        if value in (None, ""):
            raise ValueError(f"Missing column '{column}' in {csv_path}.")
        values.append(float(value))
    return np.asarray(values, dtype=float)


def _load_summary_series(csv_path: str | Path) -> dict[str, np.ndarray]:
    rows = _time_summary_rows(csv_path)
    return {
        "time_index": _time_axis(rows),
        "sample_mean": _float_series(rows, "sample_mean", csv_path=csv_path),
        "q025": _float_series(rows, "q025", csv_path=csv_path),
        "q975": _float_series(rows, "q975", csv_path=csv_path),
    }


def _observed_time_mean(experiment_root: str | Path) -> tuple[np.ndarray, np.ndarray, int]:
    panel_context = load_experiment_panel_context(experiment_root)
    observed = np.mean(np.asarray(panel_context["x"], dtype=float), axis=1)
    time_index = np.arange(observed.shape[0], dtype=int)
    return time_index, observed, int(panel_context["s"])


def _plot_observed_trajectory(
    ax: plt.Axes,
    time_index: np.ndarray,
    observed_mean: np.ndarray,
) -> None:
    ax.plot(
        time_index,
        observed_mean,
        color="black",
        linewidth=2.5,
        label="Observed",
        zorder=5,
    )


def _plot_sample_trajectory(
    ax: plt.Axes,
    time_index: np.ndarray,
    sample_mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    label: str,
    color: object,
) -> None:
    ax.fill_between(
        time_index,
        lower,
        upper,
        color=color,
        alpha=0.18,
        linewidth=0.0,
    )
    ax.plot(
        time_index,
        sample_mean,
        color=color,
        linewidth=2.0,
        label=label,
        zorder=3,
    )


def _finalize_axis(
    ax: plt.Axes,
    *,
    title: str,
    show_intervention_start: int | None,
) -> None:
    if show_intervention_start is not None:
        ax.axvline(
            int(show_intervention_start),
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
    ax.set_xlabel("Time index")
    ax.set_ylabel("Average outcome")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")


def _save_figure(fig: plt.Figure, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _resolved_roots(manifest_row: dict[str, str]) -> tuple[Path, Path]:
    experiment_root = resolve_orcd_local_path(str(manifest_row.get("experiment_path", "")))
    output_root = resolve_orcd_local_path(str(manifest_row.get("output_path", "")))
    return experiment_root, output_root


def _posterior_predictive_rows(
    manifest_rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        if str(row.get("target_intervention_source", "")).strip() != "observed_experiment":
            continue
        if str(row.get("source_type", "")).strip() != "fit":
            continue
        experiment_root, _ = _resolved_roots(row)
        grouped[(str(experiment_root), str(row.get("run_slug", "")))].append(row)
    return grouped


def _intervention_rows(
    manifest_rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        if str(row.get("target_intervention_source", "")).strip() != "saved_intervention":
            continue
        if str(row.get("source_type", "")).strip() != "fit":
            continue
        experiment_root, _ = _resolved_roots(row)
        grouped[
            (
                str(experiment_root),
                str(row.get("run_slug", "")),
                str(row.get("source_slug", "")),
            )
        ].append(row)
    return grouped


def _write_posterior_predictive_plots(
    manifest_rows: list[dict[str, str]],
    *,
    output_dir_name: str,
) -> tuple[list[str], list[str]]:
    grouped = _posterior_predictive_rows(manifest_rows)
    if not grouped:
        return [], ["Skipped posterior-predictive plots: no eligible fit rows found."]

    output_paths: list[str] = []
    colors = list(plt.get_cmap("tab10").colors)
    for (experiment_root_text, run_slug), group_rows in sorted(grouped.items()):
        experiment_root = Path(experiment_root_text)
        time_index, observed_mean, intervention_start = _observed_time_mean(experiment_root)
        fig, ax = plt.subplots(figsize=(11, 6))
        _plot_observed_trajectory(ax, time_index, observed_mean)
        for color_index, row in enumerate(
            sorted(group_rows, key=lambda item: str(item.get("source_name", "")))
        ):
            _, output_root = _resolved_roots(row)
            series = _load_summary_series(output_root / "posterior_predictive_time_summary.csv")
            _plot_sample_trajectory(
                ax,
                series["time_index"],
                series["sample_mean"],
                series["q025"],
                series["q975"],
                label=str(row.get("source_name", row.get("source_slug", ""))),
                color=colors[color_index % len(colors)],
            )
        title = (
            f"Posterior predictive average outcome over time"
            f" ({group_rows[0].get('experiment_name', '')}, {group_rows[0].get('run_name', '')})"
        )
        _finalize_axis(
            ax,
            title=title,
            show_intervention_start=intervention_start,
        )
        output_path = (
            experiment_root
            / output_dir_name
            / "posterior_predictive"
            / f"{run_slug}_time_mean.png"
        )
        output_paths.append(_save_figure(fig, output_path))
    return output_paths, []


def _write_intervention_summary_plots(
    manifest_rows: list[dict[str, str]],
    *,
    output_dir_name: str,
) -> tuple[list[str], list[str]]:
    grouped = _intervention_rows(manifest_rows)
    if not grouped:
        return [], ["Skipped intervention-summary plots: no eligible fit rows found."]

    output_paths: list[str] = []
    colors = list(plt.get_cmap("tab10").colors)
    for (experiment_root_text, run_slug, source_slug), group_rows in sorted(grouped.items()):
        experiment_root = Path(experiment_root_text)
        time_index, observed_mean, _ = _observed_time_mean(experiment_root)
        fig, ax = plt.subplots(figsize=(11, 6))
        _plot_observed_trajectory(ax, time_index, observed_mean)
        for color_index, row in enumerate(
            sorted(
                group_rows,
                key=lambda item: str(
                    item.get("target_intervention_name", item.get("target_intervention_slug", ""))
                ),
            )
        ):
            _, output_root = _resolved_roots(row)
            series = _load_summary_series(output_root / "counterfactual_time_summary.csv")
            _plot_sample_trajectory(
                ax,
                series["time_index"],
                series["sample_mean"],
                series["q025"],
                series["q975"],
                label=str(
                    row.get("target_intervention_name", row.get("target_intervention_slug", ""))
                ),
                color=colors[color_index % len(colors)],
            )
        title = (
            f"Intervention average outcome over time"
            f" ({group_rows[0].get('experiment_name', '')}, {group_rows[0].get('source_name', '')}, {group_rows[0].get('run_name', '')})"
        )
        _finalize_axis(
            ax,
            title=title,
            show_intervention_start=None,
        )
        output_path = (
            experiment_root
            / output_dir_name
            / "intervention_summaries"
            / f"{run_slug}__{source_slug}_time_mean.png"
        )
        output_paths.append(_save_figure(fig, output_path))
    return output_paths, []


def write_posterior_predictive_plot_reports(
    manifest_path: str | Path,
    *,
    plot_posterior_predictive: bool,
    plot_intervention_summaries: bool,
    output_dir_name: str = "posterior_predictive_reports",
) -> dict[str, object]:
    manifest_rows = read_csv_rows(manifest_path)
    if not manifest_rows:
        raise ValueError(f"No rows found in posterior-predictive manifest {manifest_path}.")

    outputs: dict[str, object] = {
        "manifest_path": str(Path(manifest_path).resolve()),
        "posterior_predictive_plot_paths": [],
        "intervention_summary_plot_paths": [],
        "messages": [],
    }
    messages: list[str] = []
    if plot_posterior_predictive:
        paths, path_messages = _write_posterior_predictive_plots(
            manifest_rows,
            output_dir_name=output_dir_name,
        )
        outputs["posterior_predictive_plot_paths"] = paths
        messages.extend(path_messages)
    if plot_intervention_summaries:
        paths, path_messages = _write_intervention_summary_plots(
            manifest_rows,
            output_dir_name=output_dir_name,
        )
        outputs["intervention_summary_plot_paths"] = paths
        messages.extend(path_messages)
    outputs["num_posterior_predictive_plots"] = len(
        outputs["posterior_predictive_plot_paths"]
    )
    outputs["num_intervention_summary_plots"] = len(
        outputs["intervention_summary_plot_paths"]
    )
    outputs["messages"] = messages
    return outputs
