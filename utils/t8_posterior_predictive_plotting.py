"""Plot posterior-predictive and intervention summary figures."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils.t0_csv_utils import read_csv_rows
from utils.t5_experiment_context import load_experiment_panel_context
from utils.t6_intervention_utils import resolve_intervention_context
from utils.t8_posterior_predictive_reporting import (
    COUNTERFACTUAL_SUMMARY_ROOT_NAME,
    POSTERIOR_PREDICTIVE_SUMMARY_ROOT_NAME,
)

_COUNTERFACTUAL_S_PRIORITY = (
    "all_intervention_from_s",
    "all_intervention",
)
_OBSERVED_COLOR = "black"
_FIT_INTERFERENCE_COLOR = "#57B271"
_FIT_NO_INTERFERENCE_COLOR = "#9B77BD"
_ALL_INTERVENTION_COLOR = "#1f77b4"
_NO_INTERVENTION_COLOR = "#ff7f0e"
_FALLBACK_COLORS = list(plt.get_cmap("tab10").colors)
_LEGEND_FONT_SIZE = 14


def _summary_rows(csv_path: str | Path, *, kind: str) -> list[dict[str, str]]:
    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No {kind} rows found in {csv_path}.")
    return rows


def _summary_axis(rows: list[dict[str, str]], index_column: str) -> np.ndarray:
    axis: list[int] = []
    for index, row in enumerate(rows):
        value = row.get(index_column, "")
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


def _load_summary_series(
    csv_path: str | Path,
    *,
    index_column: str,
    kind: str,
) -> dict[str, np.ndarray]:
    rows = _summary_rows(csv_path, kind=kind)
    return {
        "index": _summary_axis(rows, index_column),
        "sample_mean": _float_series(rows, "sample_mean", csv_path=csv_path),
        "q025": _float_series(rows, "q025", csv_path=csv_path),
        "q975": _float_series(rows, "q975", csv_path=csv_path),
    }


def _intervention_share_series(z: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(z, dtype=float) == 1.0, axis=1)


def _load_observed_panel_context(experiment_root: str | Path) -> dict[str, object]:
    return load_experiment_panel_context(experiment_root)


def _observed_time_mean(
    panel_context: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, int]:
    observed = np.mean(np.asarray(panel_context["x"], dtype=float), axis=1)
    time_index = np.arange(observed.shape[0], dtype=int)
    return time_index, observed, int(panel_context["s"])


def _observed_unit_mean(
    panel_context: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.mean(np.asarray(panel_context["x"], dtype=float), axis=0)
    unit_index = np.arange(observed.shape[0], dtype=int)
    return unit_index, observed


def _unit_order_from_observed(observed_mean: np.ndarray) -> np.ndarray:
    return np.argsort(np.asarray(observed_mean, dtype=float), kind="stable")


def _apply_unit_order(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)[np.asarray(order, dtype=int)]


def _plot_observed_trajectory(
    ax: plt.Axes,
    index_axis: np.ndarray,
    observed_mean: np.ndarray,
) -> None:
    ax.plot(
        index_axis,
        observed_mean,
        color=_OBSERVED_COLOR,
        linewidth=2.5,
        label="Observed",
        zorder=5,
    )


def _plot_sample_trajectory(
    ax: plt.Axes,
    index_axis: np.ndarray,
    sample_mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    label: str,
    color: object,
) -> None:
    ax.fill_between(
        index_axis,
        lower,
        upper,
        color=color,
        alpha=0.18,
        linewidth=0.0,
    )
    ax.plot(
        index_axis,
        sample_mean,
        color=color,
        linewidth=2.0,
        label=label,
        zorder=3,
    )


def _plot_line_trajectory(
    ax: plt.Axes,
    index_axis: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: object,
) -> None:
    ax.plot(
        index_axis,
        values,
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
    x_label: str,
    y_label: str,
) -> None:
    if show_intervention_start is not None:
        ax.axvline(
            int(show_intervention_start),
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=_LEGEND_FONT_SIZE)


def _contains_xi_zero(*values: str) -> bool:
    return any("xi_zero" in str(value).strip() for value in values)


def _posterior_predictive_label(row: dict[str, str], *, pretty: bool) -> str:
    source_name = str(row.get("source_name", "")).strip()
    source_slug = str(row.get("source_slug", "")).strip()
    if pretty:
        if _contains_xi_zero(source_name, source_slug):
            return "No Interference"
        return "Interference"
    return source_name or source_slug


def _posterior_predictive_color(row: dict[str, str]) -> str:
    source_name = str(row.get("source_name", "")).strip()
    source_slug = str(row.get("source_slug", "")).strip()
    if _contains_xi_zero(source_name, source_slug):
        return _FIT_NO_INTERFERENCE_COLOR
    return _FIT_INTERFERENCE_COLOR


def _counterfactual_label(row: dict[str, str], *, pretty: bool) -> str:
    intervention_slug = str(row.get("target_intervention_slug", "")).strip()
    intervention_name = str(
        row.get("target_intervention_name", row.get("target_intervention_slug", ""))
    ).strip()
    if pretty:
        if intervention_slug == "all_intervention_from_s":
            return "All Intervention"
        if intervention_slug == "no_intervention":
            return "No Intervention"
    return intervention_name or intervention_slug


def _counterfactual_color(
    row: dict[str, str],
    *,
    fallback_index: int,
) -> object:
    intervention_slug = str(row.get("target_intervention_slug", "")).strip()
    if intervention_slug in {"all_intervention_from_s", "all_intervention"}:
        return _ALL_INTERVENTION_COLOR
    if intervention_slug == "no_intervention":
        return _NO_INTERVENTION_COLOR
    return _FALLBACK_COLORS[fallback_index % len(_FALLBACK_COLORS)]


def _observed_intervention_label(*, pretty: bool) -> str:
    if pretty:
        return "Observed Intervention"
    return "observed_experiment"


def _save_figure(fig: plt.Figure, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _resolved_roots(manifest_row: dict[str, str]) -> tuple[Path, Path]:
    experiment_root = Path(str(manifest_row.get("experiment_path", ""))).resolve()
    output_root = Path(str(manifest_row.get("output_path", ""))).resolve()
    return experiment_root, output_root


def _report_root(experiment_root: Path, output_dir_name: str) -> Path:
    if not str(output_dir_name).strip():
        return experiment_root
    return experiment_root / str(output_dir_name)


def _counterfactual_plot_s(
    experiment_root: Path,
    group_rows: list[dict[str, str]],
) -> int | None:
    rows_by_slug = {
        str(row.get("target_intervention_slug", "")).strip(): row for row in group_rows
    }
    for intervention_slug in _COUNTERFACTUAL_S_PRIORITY:
        row = rows_by_slug.get(intervention_slug)
        if row is None:
            continue
        intervention_name = str(
            row.get("target_intervention_name", row.get("target_intervention_slug", ""))
        ).strip()
        if not intervention_name:
            continue
        context = resolve_intervention_context(
            experiment_root,
            intervention_source="saved_intervention",
            intervention_name=intervention_name,
        )
        return int(context.s)
    return None


def _posterior_predictive_rows(
    manifest_rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        if (
            str(row.get("target_intervention_source", "")).strip()
            != "observed_experiment"
        ):
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
        if (
            str(row.get("target_intervention_source", "")).strip()
            != "saved_intervention"
        ):
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


def _intervention_share_rows_by_experiment(
    manifest_rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_experiments: set[str] = set()
    for row in manifest_rows:
        intervention_source = str(row.get("target_intervention_source", "")).strip()
        if intervention_source != "observed_experiment":
            continue
        experiment_root, _ = _resolved_roots(row)
        experiment_key = str(experiment_root)
        if experiment_key in seen_experiments:
            continue
        seen_experiments.add(experiment_key)
        grouped[experiment_key].append(row)
    return grouped


def _counterfactual_group_sort_key(item: dict[str, str]) -> str:
    return str(
        item.get("target_intervention_name", item.get("target_intervention_slug", ""))
    )


def _write_posterior_predictive_time_plots(
    manifest_rows: list[dict[str, str]],
    *,
    output_dir_name: str,
    pretty: bool,
) -> tuple[list[str], list[str]]:
    grouped = _posterior_predictive_rows(manifest_rows)
    if not grouped:
        return [], ["Skipped posterior-predictive plots: no eligible fit rows found."]

    output_paths: list[str] = []
    for (experiment_root_text, run_slug), group_rows in sorted(grouped.items()):
        experiment_root = Path(experiment_root_text)
        panel_context = _load_observed_panel_context(experiment_root)
        time_index, observed_mean, intervention_start = _observed_time_mean(
            panel_context
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        _plot_observed_trajectory(ax, time_index, observed_mean)
        for row in sorted(
            group_rows, key=lambda item: str(item.get("source_name", ""))
        ):
            _, output_root = _resolved_roots(row)
            series = _load_summary_series(
                output_root / "posterior_predictive_time_summary.csv",
                index_column="time_index",
                kind="time-summary",
            )
            _plot_sample_trajectory(
                ax,
                series["index"],
                series["sample_mean"],
                series["q025"],
                series["q975"],
                label=_posterior_predictive_label(row, pretty=pretty),
                color=_posterior_predictive_color(row),
            )
        _finalize_axis(
            ax,
            title="",
            show_intervention_start=intervention_start,
            x_label="Time index",
            y_label="Average outcome",
        )
        output_path = (
            _report_root(experiment_root, output_dir_name)
            / POSTERIOR_PREDICTIVE_SUMMARY_ROOT_NAME
            / f"{run_slug}_time_mean.png"
        )
        output_paths.append(_save_figure(fig, output_path))
    return output_paths, []


def _write_posterior_predictive_unit_plots(
    manifest_rows: list[dict[str, str]],
    *,
    output_dir_name: str,
    pretty: bool,
) -> tuple[list[str], list[str]]:
    grouped = _posterior_predictive_rows(manifest_rows)
    if not grouped:
        return [], [
            "Skipped posterior-predictive unit plots: no eligible fit rows found."
        ]

    output_paths: list[str] = []
    for (experiment_root_text, run_slug), group_rows in sorted(grouped.items()):
        experiment_root = Path(experiment_root_text)
        panel_context = _load_observed_panel_context(experiment_root)
        unit_index, observed_mean = _observed_unit_mean(panel_context)
        unit_order = _unit_order_from_observed(observed_mean)
        unit_index = np.arange(observed_mean.shape[0], dtype=int)
        observed_mean = _apply_unit_order(observed_mean, unit_order)
        fig, ax = plt.subplots(figsize=(11, 6))
        _plot_observed_trajectory(ax, unit_index, observed_mean)
        for row in sorted(
            group_rows, key=lambda item: str(item.get("source_name", ""))
        ):
            _, output_root = _resolved_roots(row)
            series = _load_summary_series(
                output_root / "posterior_predictive_unit_summary.csv",
                index_column="unit_index",
                kind="unit-summary",
            )
            _plot_sample_trajectory(
                ax,
                unit_index,
                _apply_unit_order(series["sample_mean"], unit_order),
                _apply_unit_order(series["q025"], unit_order),
                _apply_unit_order(series["q975"], unit_order),
                label=_posterior_predictive_label(row, pretty=pretty),
                color=_posterior_predictive_color(row),
            )
        _finalize_axis(
            ax,
            title="",
            show_intervention_start=None,
            x_label="Unit rank (sorted by observed outcome)",
            y_label="Average outcome",
        )
        output_path = (
            _report_root(experiment_root, output_dir_name)
            / POSTERIOR_PREDICTIVE_SUMMARY_ROOT_NAME
            / f"{run_slug}_unit_mean.png"
        )
        output_paths.append(_save_figure(fig, output_path))
    return output_paths, []


def _write_intervention_summary_time_plots(
    manifest_rows: list[dict[str, str]],
    *,
    output_dir_name: str,
    pretty: bool,
) -> tuple[list[str], list[str], list[str]]:
    grouped = _intervention_rows(manifest_rows)
    if not grouped:
        return (
            [],
            [],
            ["Skipped intervention-summary plots: no eligible fit rows found."],
        )

    output_paths: list[str] = []
    post_s_output_paths: list[str] = []
    for (experiment_root_text, run_slug, source_slug), group_rows in sorted(
        grouped.items()
    ):
        experiment_root = Path(experiment_root_text)
        panel_context = _load_observed_panel_context(experiment_root)
        time_index, observed_mean, _ = _observed_time_mean(panel_context)
        intervention_start = _counterfactual_plot_s(experiment_root, group_rows)

        fig, ax = plt.subplots(figsize=(11, 6))
        _plot_observed_trajectory(ax, time_index, observed_mean)
        sorted_rows = sorted(group_rows, key=_counterfactual_group_sort_key)
        for color_index, row in enumerate(sorted_rows):
            _, output_root = _resolved_roots(row)
            series = _load_summary_series(
                output_root / "counterfactual_time_summary.csv",
                index_column="time_index",
                kind="time-summary",
            )
            _plot_sample_trajectory(
                ax,
                series["index"],
                series["sample_mean"],
                series["q025"],
                series["q975"],
                label=_counterfactual_label(row, pretty=pretty),
                color=_counterfactual_color(row, fallback_index=color_index),
            )
        _finalize_axis(
            ax,
            title="",
            show_intervention_start=intervention_start,
            x_label="Time index",
            y_label="Average outcome",
        )
        output_path = (
            _report_root(experiment_root, output_dir_name)
            / COUNTERFACTUAL_SUMMARY_ROOT_NAME
            / f"{run_slug}__{source_slug}_time_mean.png"
        )
        output_paths.append(_save_figure(fig, output_path))

        if intervention_start is None:
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        _plot_observed_trajectory(ax, time_index, observed_mean)
        for color_index, row in enumerate(sorted_rows):
            _, output_root = _resolved_roots(row)
            series = _load_summary_series(
                output_root / "counterfactual_time_summary.csv",
                index_column="time_index",
                kind="time-summary",
            )
            keep = series["index"] >= int(intervention_start)
            _plot_sample_trajectory(
                ax,
                series["index"][keep],
                series["sample_mean"][keep],
                series["q025"][keep],
                series["q975"][keep],
                label=_counterfactual_label(row, pretty=pretty),
                color=_counterfactual_color(row, fallback_index=color_index),
            )
        _finalize_axis(
            ax,
            title="",
            show_intervention_start=intervention_start,
            x_label="Time index",
            y_label="Average outcome",
        )
        output_path = (
            _report_root(experiment_root, output_dir_name)
            / COUNTERFACTUAL_SUMMARY_ROOT_NAME
            / f"{run_slug}__{source_slug}_time_mean_post_s.png"
        )
        post_s_output_paths.append(_save_figure(fig, output_path))
    return output_paths, post_s_output_paths, []


def _write_intervention_summary_unit_plots(
    manifest_rows: list[dict[str, str]],
    *,
    output_dir_name: str,
    pretty: bool,
) -> tuple[list[str], list[str]]:
    grouped = _intervention_rows(manifest_rows)
    if not grouped:
        return [], [
            "Skipped intervention-summary unit plots: no eligible fit rows found."
        ]

    output_paths: list[str] = []
    for (experiment_root_text, run_slug, source_slug), group_rows in sorted(
        grouped.items()
    ):
        experiment_root = Path(experiment_root_text)
        panel_context = _load_observed_panel_context(experiment_root)
        unit_index, observed_mean = _observed_unit_mean(panel_context)
        unit_order = _unit_order_from_observed(observed_mean)
        unit_index = np.arange(observed_mean.shape[0], dtype=int)
        observed_mean = _apply_unit_order(observed_mean, unit_order)
        fig, ax = plt.subplots(figsize=(11, 6))
        _plot_observed_trajectory(ax, unit_index, observed_mean)
        for color_index, row in enumerate(
            sorted(group_rows, key=_counterfactual_group_sort_key)
        ):
            _, output_root = _resolved_roots(row)
            series = _load_summary_series(
                output_root / "counterfactual_unit_summary.csv",
                index_column="unit_index",
                kind="unit-summary",
            )
            _plot_sample_trajectory(
                ax,
                unit_index,
                _apply_unit_order(series["sample_mean"], unit_order),
                _apply_unit_order(series["q025"], unit_order),
                _apply_unit_order(series["q975"], unit_order),
                label=_counterfactual_label(row, pretty=pretty),
                color=_counterfactual_color(row, fallback_index=color_index),
            )
        _finalize_axis(
            ax,
            title="",
            show_intervention_start=None,
            x_label="Unit rank (sorted by observed outcome)",
            y_label="Average outcome",
        )
        output_path = (
            _report_root(experiment_root, output_dir_name)
            / COUNTERFACTUAL_SUMMARY_ROOT_NAME
            / f"{run_slug}__{source_slug}_unit_mean.png"
        )
        output_paths.append(_save_figure(fig, output_path))
    return output_paths, []


def _write_intervention_share_plots(
    manifest_rows: list[dict[str, str]],
    *,
    output_dir_name: str,
    pretty: bool,
) -> tuple[list[str], list[str]]:
    grouped = _intervention_share_rows_by_experiment(manifest_rows)
    if not grouped:
        return [], ["Skipped intervention-share plots: no intervention rows found."]

    output_paths: list[str] = []
    for experiment_root_text, group_rows in sorted(grouped.items()):
        experiment_root = Path(experiment_root_text)
        fig, ax = plt.subplots(figsize=(11, 5))
        row = group_rows[0]
        context = resolve_intervention_context(
            experiment_root,
            intervention_source="observed_experiment",
            intervention_name="",
        )
        share_series = _intervention_share_series(context.z)
        time_index = np.arange(share_series.shape[0], dtype=int)
        _plot_line_trajectory(
            ax,
            time_index,
            share_series,
            label=_observed_intervention_label(pretty=pretty),
            color=_OBSERVED_COLOR,
        )
        ax.axvline(
            50,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.set_xlabel("Time index")
        ax.set_ylabel("Share intervened")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=_LEGEND_FONT_SIZE)
        output_path = (
            _report_root(experiment_root, output_dir_name)
            / COUNTERFACTUAL_SUMMARY_ROOT_NAME
            / "intervention_share_over_time.png"
        )
        output_paths.append(_save_figure(fig, output_path))
    return output_paths, []


def write_posterior_predictive_plot_reports(
    manifest_path: str | Path,
    *,
    plot_posterior_predictive: bool,
    plot_intervention_summaries: bool,
    output_dir_name: str = "",
    pretty: bool = False,
) -> dict[str, object]:
    manifest_rows = read_csv_rows(manifest_path)
    if not manifest_rows:
        raise ValueError(
            f"No rows found in posterior-predictive manifest {manifest_path}."
        )

    outputs: dict[str, object] = {
        "manifest_path": str(Path(manifest_path).resolve()),
        "posterior_predictive_plot_paths": [],
        "posterior_predictive_time_plot_paths": [],
        "posterior_predictive_unit_plot_paths": [],
        "counterfactual_summary_plot_paths": [],
        "counterfactual_summary_time_plot_paths": [],
        "counterfactual_summary_unit_plot_paths": [],
        "counterfactual_summary_post_s_plot_paths": [],
        "intervention_share_plot_paths": [],
        "messages": [],
    }
    messages: list[str] = []
    if plot_posterior_predictive:
        time_paths, path_messages = _write_posterior_predictive_time_plots(
            manifest_rows,
            output_dir_name=output_dir_name,
            pretty=pretty,
        )
        unit_paths, unit_messages = _write_posterior_predictive_unit_plots(
            manifest_rows,
            output_dir_name=output_dir_name,
            pretty=pretty,
        )
        outputs["posterior_predictive_time_plot_paths"] = time_paths
        outputs["posterior_predictive_unit_plot_paths"] = unit_paths
        outputs["posterior_predictive_plot_paths"] = [*time_paths, *unit_paths]
        messages.extend(path_messages)
        messages.extend(unit_messages)
    if plot_intervention_summaries:
        time_paths, post_s_paths, path_messages = (
            _write_intervention_summary_time_plots(
                manifest_rows,
                output_dir_name=output_dir_name,
                pretty=pretty,
            )
        )
        unit_paths, unit_messages = _write_intervention_summary_unit_plots(
            manifest_rows,
            output_dir_name=output_dir_name,
            pretty=pretty,
        )
        outputs["counterfactual_summary_time_plot_paths"] = time_paths
        outputs["counterfactual_summary_unit_plot_paths"] = unit_paths
        outputs["counterfactual_summary_post_s_plot_paths"] = post_s_paths
        outputs["counterfactual_summary_plot_paths"] = [
            *time_paths,
            *unit_paths,
            *post_s_paths,
        ]
        messages.extend(path_messages)
        messages.extend(unit_messages)
        paths, path_messages = _write_intervention_share_plots(
            manifest_rows,
            output_dir_name=output_dir_name,
            pretty=pretty,
        )
        outputs["intervention_share_plot_paths"] = paths
        messages.extend(path_messages)
    outputs["num_posterior_predictive_plots"] = len(
        outputs["posterior_predictive_plot_paths"]
    )
    outputs["num_posterior_predictive_time_plots"] = len(
        outputs["posterior_predictive_time_plot_paths"]
    )
    outputs["num_posterior_predictive_unit_plots"] = len(
        outputs["posterior_predictive_unit_plot_paths"]
    )
    outputs["num_counterfactual_summary_plots"] = len(
        outputs["counterfactual_summary_plot_paths"]
    )
    outputs["num_counterfactual_summary_time_plots"] = len(
        outputs["counterfactual_summary_time_plot_paths"]
    )
    outputs["num_counterfactual_summary_unit_plots"] = len(
        outputs["counterfactual_summary_unit_plot_paths"]
    )
    outputs["num_counterfactual_summary_post_s_plots"] = len(
        outputs["counterfactual_summary_post_s_plot_paths"]
    )
    outputs["num_intervention_share_plots"] = len(
        outputs["intervention_share_plot_paths"]
    )
    outputs["intervention_summary_plot_paths"] = outputs[
        "counterfactual_summary_plot_paths"
    ]
    outputs["num_intervention_summary_plots"] = outputs[
        "num_counterfactual_summary_plots"
    ]
    outputs["messages"] = messages
    return outputs
