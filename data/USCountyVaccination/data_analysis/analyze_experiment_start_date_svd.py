"""SVD diagnostics for start-date-sliced USCountyVaccination experiment panels."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

try:
    from ..experiment_artifacts import (
        apply_optional_trim,
        build_node_table,
        build_realized_intervention_artifact,
        build_realized_outcome_artifact,
        canonical_time_index,
        load_inputs,
        select_dense_suffix_support,
    )
except ImportError:  # pragma: no cover - direct script fallback
    from experiment_artifacts import (
        apply_optional_trim,
        build_node_table,
        build_realized_intervention_artifact,
        build_realized_outcome_artifact,
        canonical_time_index,
        load_inputs,
        select_dense_suffix_support,
    )


DEFAULT_START_DATES = ["2020-03-01", "2020-06-07", "2020-09-06"]
DEFAULT_INTERVENTIONS = [
    "complete_cov_ge_20",
    "complete_cov_ge_30",
    "complete_cov_ge_40",
]
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "us_county_experiment_svd"
)

VIEW_LABELS = {
    "raw": "Raw",
    "row_centered": "Row-centered",
    "row_column_centered": "Row-and-column centered",
}


@dataclass(frozen=True)
class SlicedPanel:
    outcome_code: str
    intervention_code: str
    requested_start_date: str
    resolved_start_week_end_date: str
    start_index: int
    time_index: pd.DataFrame
    x: np.ndarray
    z: np.ndarray
    x_0: np.ndarray
    z_0: np.ndarray
    node_count: int
    transition_weeks: int
    support_metadata: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze low-rank structure in USCountyVaccination experiment panels "
            "for requested start dates."
        )
    )
    parser.add_argument(
        "--start_dates",
        nargs="*",
        default=DEFAULT_START_DATES,
        help="ISO dates resolved forward to the first available modeled WeekEndDate.",
    )
    parser.add_argument(
        "--interventions",
        nargs="*",
        default=DEFAULT_INTERVENTIONS,
        help="Intervention codes to analyze.",
    )
    parser.add_argument(
        "--outcome_code",
        default="death_rate_100k_ge_2",
        help="Outcome code to analyze.",
    )
    parser.add_argument(
        "--lag_code",
        default="2w",
        help="Lag code applied to the intervention artifact.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV and Markdown summaries will be written.",
    )
    parser.add_argument(
        "--trim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the standard mainland-US and population >= 2000 trim.",
    )
    return parser.parse_args()


def apply_view(matrix: np.ndarray, view_code: str) -> np.ndarray:
    if view_code == "raw":
        return matrix.copy()
    if view_code == "row_centered":
        return matrix - matrix.mean(axis=1, keepdims=True)
    if view_code == "row_column_centered":
        row_mean = matrix.mean(axis=1, keepdims=True)
        col_mean = matrix.mean(axis=0, keepdims=True)
        grand_mean = matrix.mean()
        return matrix - row_mean - col_mean + grand_mean
    raise ValueError(f"Unknown view_code={view_code}")


def effective_rank(cumulative_energy: np.ndarray, threshold: float) -> int:
    if cumulative_energy.size == 0:
        return 0
    index = int(np.searchsorted(cumulative_energy, threshold, side="left"))
    return min(index + 1, int(cumulative_energy.size))


def classify_low_rank(rank_90: int, rank_95: int, leading_share: float) -> str:
    if rank_90 <= 2 and rank_95 <= 4 and leading_share >= 0.50:
        return "strong"
    if rank_90 <= 6 and rank_95 <= 12 and leading_share >= 0.20:
        return "moderate"
    return "weak"


def summarize_view(matrix: np.ndarray, view_code: str) -> tuple[pd.DataFrame, dict[str, object]]:
    singular_values = np.linalg.svd(matrix, compute_uv=False, full_matrices=False)
    energy = singular_values**2
    total_energy = float(energy.sum())
    energy_share = (
        energy / total_energy if total_energy > 0.0 else np.zeros_like(energy, dtype=float)
    )
    cumulative_energy_share = np.cumsum(energy_share)
    next_ratio = np.full(singular_values.shape, np.nan, dtype=float)
    if singular_values.size >= 2:
        denominator = np.maximum(singular_values[1:], 1e-12)
        next_ratio[:-1] = singular_values[:-1] / denominator
    effective_rank_80 = effective_rank(cumulative_energy_share, 0.80)
    effective_rank_90 = effective_rank(cumulative_energy_share, 0.90)
    effective_rank_95 = effective_rank(cumulative_energy_share, 0.95)
    effective_rank_99 = effective_rank(cumulative_energy_share, 0.99)
    if singular_values.size >= 2:
        search_limit = max(1, min(singular_values.size - 1, effective_rank_90))
        elbow_rank = int(np.nanargmax(next_ratio[:search_limit]) + 1)
        elbow_ratio = float(next_ratio[elbow_rank - 1])
    else:
        elbow_rank = 1
        elbow_ratio = float("nan")

    spectrum = pd.DataFrame(
        {
            "view_code": view_code,
            "view_label": VIEW_LABELS[view_code],
            "component_rank": np.arange(1, singular_values.size + 1, dtype=int),
            "singular_value": singular_values,
            "energy_share": energy_share,
            "cumulative_energy_share": cumulative_energy_share,
            "next_singular_ratio": next_ratio,
        }
    )
    leading_share = float(energy_share[0]) if energy_share.size else float("nan")
    summary = {
        "view_code": view_code,
        "view_label": VIEW_LABELS[view_code],
        "leading_singular_value": float(singular_values[0]) if singular_values.size else float("nan"),
        "leading_energy_share": leading_share,
        "effective_rank_80": effective_rank_80,
        "effective_rank_90": effective_rank_90,
        "effective_rank_95": effective_rank_95,
        "effective_rank_99": effective_rank_99,
        "elbow_rank": elbow_rank,
        "elbow_ratio": elbow_ratio,
        "low_rank_strength": classify_low_rank(
            effective_rank_90,
            effective_rank_95,
            leading_share,
        ),
    }
    return spectrum, summary


def resolve_start_index(time_index: pd.DataFrame, start_date: str) -> tuple[int, str]:
    requested = pd.Timestamp(start_date).normalize()
    week_ends = pd.to_datetime(time_index["WeekEndDate"]).dt.normalize()
    matches = np.flatnonzero(week_ends.ge(requested).to_numpy())
    if matches.size == 0:
        raise ValueError(
            f"Start date {start_date} is after the last available week "
            f"{week_ends.max().date().isoformat()}."
        )
    start_index = int(matches[0])
    if start_index >= len(time_index) - 1:
        raise ValueError(f"Start date {start_date} leaves no transition weeks to analyze.")
    return start_index, week_ends.iloc[start_index].date().isoformat()


def slice_panel_for_start(
    x: np.ndarray,
    z: np.ndarray,
    x_0: np.ndarray,
    z_0: np.ndarray,
    time_index: pd.DataFrame,
    start_date: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int, str]:
    if len(time_index) != x.shape[0] + 1:
        raise ValueError(
            f"time_index has {len(time_index)} rows but x has {x.shape[0]} transition rows."
        )
    start_index, resolved_week_end = resolve_start_index(time_index, start_date)
    x_all = np.vstack([x_0[None, :], x])
    z_all = np.vstack([z_0[None, :], z])
    sliced_time_index = time_index.iloc[start_index:].reset_index(drop=True).copy()
    sliced_time_index["model_index"] = np.arange(len(sliced_time_index), dtype=int)
    sliced_x = x_all[start_index + 1 :].astype(np.int8)
    sliced_z = z_all[start_index + 1 :].astype(np.int8)
    sliced_x_0 = x_all[start_index].astype(np.int8)
    sliced_z_0 = z_all[start_index].astype(np.int8)
    return (
        sliced_x,
        sliced_z,
        sliced_x_0,
        sliced_z_0,
        sliced_time_index,
        start_index,
        resolved_week_end,
    )


def build_sliced_panels(
    outcome_code: str,
    intervention_codes: list[str],
    lag_code: str,
    start_dates: list[str],
    trim: bool,
) -> list[SlicedPanel]:
    panel, node_geography, centroids = load_inputs()
    full_node_table = build_node_table(node_geography, centroids)
    full_node_table, panel, _ = apply_optional_trim(full_node_table, panel, trim)
    full_node_order = full_node_table["fips"].astype(str).tolist()
    time_index = canonical_time_index(panel)

    outcome_artifact = build_realized_outcome_artifact(
        panel=panel,
        node_order=full_node_order,
        time_index=time_index,
        outcome_code=outcome_code,
        trim_applied=trim,
        artifact_dir=Path("."),
    )

    sliced_panels: list[SlicedPanel] = []
    for intervention_code in intervention_codes:
        intervention_artifact = build_realized_intervention_artifact(
            panel=panel,
            node_order=full_node_order,
            time_index=time_index,
            intervention_code=intervention_code,
            lag_code=lag_code,
            trim_applied=trim,
            artifact_dir=Path("."),
        )
        x, z, x_0, z_0, shared_time_index, realized_node_order, support_metadata = (
            select_dense_suffix_support(outcome_artifact, intervention_artifact)
        )
        for start_date in start_dates:
            (
                sliced_x,
                sliced_z,
                sliced_x_0,
                sliced_z_0,
                sliced_time_index,
                start_index,
                resolved_week_end,
            ) = slice_panel_for_start(
                x,
                z,
                x_0,
                z_0,
                shared_time_index,
                start_date,
            )
            sliced_panels.append(
                SlicedPanel(
                    outcome_code=outcome_code,
                    intervention_code=intervention_code,
                    requested_start_date=pd.Timestamp(start_date).date().isoformat(),
                    resolved_start_week_end_date=resolved_week_end,
                    start_index=start_index,
                    time_index=sliced_time_index,
                    x=sliced_x,
                    z=sliced_z,
                    x_0=sliced_x_0,
                    z_0=sliced_z_0,
                    node_count=len(realized_node_order),
                    transition_weeks=int(sliced_x.shape[0]),
                    support_metadata=dict(support_metadata),
                )
            )
    return sliced_panels


def analyze_matrix(
    matrix: np.ndarray,
    matrix_kind: str,
    sliced_panel: SlicedPanel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectra: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for view_code in VIEW_LABELS:
        transformed = apply_view(matrix, view_code)
        spectrum, summary = summarize_view(transformed, view_code)
        spectrum["matrix_kind"] = matrix_kind
        spectrum["outcome_code"] = sliced_panel.outcome_code
        spectrum["intervention_code"] = sliced_panel.intervention_code
        spectrum["requested_start_date"] = sliced_panel.requested_start_date
        spectrum["resolved_start_week_end_date"] = sliced_panel.resolved_start_week_end_date
        spectrum["start_index"] = sliced_panel.start_index
        spectrum["node_count"] = sliced_panel.node_count
        spectrum["transition_weeks"] = sliced_panel.transition_weeks
        spectra.append(spectrum)

        summary.update(
            {
                "matrix_kind": matrix_kind,
                "outcome_code": sliced_panel.outcome_code,
                "intervention_code": sliced_panel.intervention_code,
                "requested_start_date": sliced_panel.requested_start_date,
                "resolved_start_week_end_date": sliced_panel.resolved_start_week_end_date,
                "start_index": sliced_panel.start_index,
                "node_count": sliced_panel.node_count,
                "transition_weeks": sliced_panel.transition_weeks,
                "initial_week_end_date": sliced_panel.time_index["WeekEndDate"].iloc[0]
                .date()
                .isoformat(),
                "final_week_end_date": sliced_panel.time_index["WeekEndDate"].iloc[-1]
                .date()
                .isoformat(),
                "requested_node_count": int(
                    sliced_panel.support_metadata["requested_node_count"]
                ),
                "shared_panel_weeks": int(
                    sliced_panel.support_metadata["realized_calendar_weeks"]
                ),
            }
        )
        summaries.append(summary)

    return pd.concat(spectra, ignore_index=True), pd.DataFrame(summaries)


def format_value(value: float) -> str:
    return f"{value:.4f}"


def build_summary_markdown(
    intervention_summary: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    output_dir: Path,
) -> str:
    lines = [
        "# US County Experiment Start-Date SVD Analysis",
        "",
        "This summary analyzes the saved experiment transition panels `z` and `x` after applying the standard trimmed support rule, the `2w` intervention lag, dense-suffix support selection, and then the requested start-date slicing.",
        "",
        "## Files",
        "",
        f"- Intervention summaries: `{(output_dir / 'intervention_svd_summary.csv').name}`",
        f"- Intervention spectra: `{(output_dir / 'intervention_svd_spectra.csv').name}`",
        f"- Outcome summaries: `{(output_dir / 'outcome_svd_summary.csv').name}`",
        f"- Outcome spectra: `{(output_dir / 'outcome_svd_spectra.csv').name}`",
        "",
        "## Intervention Highlights",
        "",
    ]
    for _, row in (
        intervention_summary.loc[intervention_summary["view_code"].eq("row_column_centered")]
        .sort_values(["intervention_code", "requested_start_date"])
        .iterrows()
    ):
        lines.append(
            f"- `{row['intervention_code']}` from `{row['requested_start_date']}` "
            f"(resolved `{row['resolved_start_week_end_date']}`): "
            f"`{row['low_rank_strength']}` low-rank evidence after row/column centering; "
            f"top component energy `{format_value(float(row['leading_energy_share']))}`, "
            f"`95%` rank `{int(row['effective_rank_95'])}`."
        )
    lines.extend(["", "## Outcome Highlights", ""])
    for _, row in (
        outcome_summary.loc[outcome_summary["view_code"].eq("row_column_centered")]
        .sort_values(["intervention_code", "requested_start_date"])
        .iterrows()
    ):
        lines.append(
            f"- Support from `{row['intervention_code']}`, start `{row['requested_start_date']}` "
            f"(resolved `{row['resolved_start_week_end_date']}`): "
            f"`{row['low_rank_strength']}` low-rank evidence after row/column centering; "
            f"top component energy `{format_value(float(row['leading_energy_share']))}`, "
            f"`95%` rank `{int(row['effective_rank_95'])}`."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sliced_panels = build_sliced_panels(
        outcome_code=str(args.outcome_code),
        intervention_codes=[str(code) for code in args.interventions],
        lag_code=str(args.lag_code),
        start_dates=[str(value) for value in args.start_dates],
        trim=bool(args.trim),
    )

    intervention_spectra: list[pd.DataFrame] = []
    intervention_summaries: list[pd.DataFrame] = []
    outcome_spectra: list[pd.DataFrame] = []
    outcome_summaries: list[pd.DataFrame] = []
    for sliced_panel in sliced_panels:
        spectrum, summary = analyze_matrix(
            sliced_panel.z.astype(float),
            "intervention_z",
            sliced_panel,
        )
        intervention_spectra.append(spectrum)
        intervention_summaries.append(summary)

        spectrum, summary = analyze_matrix(
            sliced_panel.x.astype(float),
            "outcome_x",
            sliced_panel,
        )
        outcome_spectra.append(spectrum)
        outcome_summaries.append(summary)

    intervention_spectra_frame = pd.concat(intervention_spectra, ignore_index=True)
    intervention_summary_frame = pd.concat(intervention_summaries, ignore_index=True)
    outcome_spectra_frame = pd.concat(outcome_spectra, ignore_index=True)
    outcome_summary_frame = pd.concat(outcome_summaries, ignore_index=True)

    intervention_spectra_frame.to_csv(
        output_dir / "intervention_svd_spectra.csv",
        index=False,
    )
    intervention_summary_frame.to_csv(
        output_dir / "intervention_svd_summary.csv",
        index=False,
    )
    outcome_spectra_frame.to_csv(
        output_dir / "outcome_svd_spectra.csv",
        index=False,
    )
    outcome_summary_frame.to_csv(
        output_dir / "outcome_svd_summary.csv",
        index=False,
    )
    (output_dir / "summary.md").write_text(
        build_summary_markdown(
            intervention_summary=intervention_summary_frame,
            outcome_summary=outcome_summary_frame,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
