"""Analyze pre-vaccination US county outcomes for low-rank spectral structure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "data_analysis" / "pre_vaccination_low_rank"
WEEKLY_PANEL_PATH = PROCESSED_DIR / "us_county_weekly_panel.csv.gz"
BINARY_PANEL_PATH = PROCESSED_DIR / "us_county_binary_panel.csv.gz"
DEFAULT_TRIMMED_NODE_INDEX_PATH = (
    BASE_DIR.parent.parent
    / "experiments"
    / "USCountyVaccination_US_trimmed"
    / "realized_outcomes"
    / "outcome_death_rate_100k_ge_2__scope_trimmed"
    / "node_index.csv"
)

DEFAULT_CUTOFF_DATE = pd.Timestamp("2020-12-27")
CONTINUOUS_COLUMN = "death_rate_100k"
BINARY_COLUMN = "x_death_rate_100k_ge_2_pm1"
VACCINATION_COLUMN = "complete_cov"

VIEW_LABELS = {
    "raw": "Raw",
    "row_centered": "Row-centered",
    "row_column_centered": "Row-and-column centered",
}


@dataclass(frozen=True)
class MatrixBundle:
    matrix_code: str
    display_name: str
    value_column: str
    matrix: np.ndarray
    weeks: list[pd.Timestamp]
    counties: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pre-vaccination low-rank structure in US county outcome panels."
    )
    parser.add_argument(
        "--cutoff_date",
        default=DEFAULT_CUTOFF_DATE.date().isoformat(),
        help="Keep rows with WeekEndDate strictly before this ISO date.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUT_DIR),
        help="Directory where summary tables and plots will be written.",
    )
    parser.add_argument(
        "--node_index_path",
        default=None,
        help="Optional node_index.csv path used to restrict the analysis to a fixed county set.",
    )
    parser.add_argument(
        "--scope_label",
        default="full",
        help="Human-readable scope label used in the markdown summary.",
    )
    return parser.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.name}. Run the USCountyVaccination preprocessing first."
        )
    return pd.read_csv(
        path,
        dtype={"fips": str},
        parse_dates=["WeekEndDate", "WeekStartDate"],
    ).sort_values(["WeekEndDate", "fips"])


def first_positive_vaccination_date(panel: pd.DataFrame) -> pd.Timestamp:
    positive_dates = panel.loc[
        pd.to_numeric(panel[VACCINATION_COLUMN], errors="coerce").fillna(0.0) > 0.0,
        "WeekEndDate",
    ]
    if positive_dates.empty:
        raise ValueError("No strictly positive vaccination coverage was found.")
    return pd.Timestamp(positive_dates.min())


def load_allowed_fips(node_index_path: Path) -> list[str]:
    if not node_index_path.exists():
        raise FileNotFoundError(
            f"Could not find node index at {node_index_path}. Rebuild the trimmed USCounty artifacts first."
        )
    node_index = pd.read_csv(node_index_path, dtype={"fips": str})
    if "fips" not in node_index.columns:
        raise ValueError(f"{node_index_path} does not contain a 'fips' column.")
    return node_index["fips"].dropna().astype(str).tolist()


def restrict_panel_to_fips(panel: pd.DataFrame, allowed_fips: list[str]) -> pd.DataFrame:
    allowed = set(allowed_fips)
    subset = panel.loc[panel["fips"].isin(allowed)].copy()
    return subset.sort_values(["WeekEndDate", "fips"]).reset_index(drop=True)


def build_matrix_bundle(
    panel: pd.DataFrame,
    matrix_code: str,
    display_name: str,
    value_column: str,
    cutoff_date: pd.Timestamp,
) -> MatrixBundle:
    subset = panel.loc[panel["WeekEndDate"] < cutoff_date].copy()
    pivot = (
        subset.pivot(index="WeekEndDate", columns="fips", values=value_column)
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.isna().any().any():
        raise ValueError(
            f"Expected a rectangular pre-vaccination matrix for {value_column}, found missing values."
        )
    matrix = pivot.to_numpy(dtype=float)
    return MatrixBundle(
        matrix_code=matrix_code,
        display_name=display_name,
        value_column=value_column,
        matrix=matrix,
        weeks=[pd.Timestamp(ts) for ts in pivot.index],
        counties=[str(fips) for fips in pivot.columns],
    )


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


def summarize_view(
    matrix_code: str,
    display_name: str,
    matrix: np.ndarray,
    view_code: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
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
            "matrix_code": matrix_code,
            "matrix_display_name": display_name,
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
        "matrix_code": matrix_code,
        "matrix_display_name": display_name,
        "view_code": view_code,
        "view_label": VIEW_LABELS[view_code],
        "n_weeks": int(matrix.shape[0]),
        "n_counties": int(matrix.shape[1]),
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


def render_scree_plot(spectrum: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for view_code, group in spectrum.groupby("view_code", sort=False):
        ax.plot(
            group["component_rank"],
            group["singular_value"],
            marker="o",
            linewidth=2.0,
            markersize=3.0,
            label=VIEW_LABELS[view_code],
        )
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Singular value")
    ax.set_title(f"{spectrum['matrix_display_name'].iloc[0]} scree plot")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_cumulative_plot(spectrum: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for view_code, group in spectrum.groupby("view_code", sort=False):
        ax.plot(
            group["component_rank"],
            group["cumulative_energy_share"],
            marker="o",
            linewidth=2.0,
            markersize=3.0,
            label=VIEW_LABELS[view_code],
        )
    for threshold in (0.80, 0.90, 0.95, 0.99):
        ax.axhline(threshold, color="#bbbbbb", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Cumulative spectral energy share")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"{spectrum['matrix_display_name'].iloc[0]} cumulative energy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_summary_markdown(
    cutoff_date: pd.Timestamp,
    first_positive_date: pd.Timestamp,
    first_nonnull_date: pd.Timestamp,
    scope_label: str,
    bundles: list[MatrixBundle],
    comparison_frames: list[pd.DataFrame],
) -> str:
    lines = [
        "# Pre-Vaccination Low-Rank Spectral Analysis",
        "",
        "This analysis isolates the nationwide county-week death outcome panel before the start of vaccination and evaluates whether the observed outcome matrix looks empirically low rank under simple spectral diagnostics.",
        "",
        "## Window",
        "",
        f"- Scope: `{scope_label}`",
        f"- Pre-vaccination cutoff: `WeekEndDate < {cutoff_date.date().isoformat()}`",
        f"- First non-missing complete vaccination coverage date in processed panel: `{first_nonnull_date.date().isoformat()}`",
        f"- First strictly positive complete vaccination coverage date in processed panel: `{first_positive_date.date().isoformat()}`",
        "",
        "## Matrices analyzed",
        "",
    ]
    for bundle in bundles:
        lines.append(
            f"- `{bundle.display_name}`: `{bundle.matrix.shape[0]}` weeks x `{bundle.matrix.shape[1]}` counties from column `{bundle.value_column}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "- `Raw` includes level effects and common weekly shocks.",
            "- `Row-centered` removes week-level means, so it downweights aggregate national waves.",
            "- `Row-and-column centered` removes both week means and county means, so any remaining concentration is more suggestive of structured low-rank dependence rather than baseline levels.",
            "- Low-rank labels here are descriptive: `strong`, `moderate`, and `weak` are based on spectral concentration, not on an exact-rank claim.",
            "",
            "## Findings",
            "",
        ]
    )
    for comparison in comparison_frames:
        display_name = str(comparison["matrix_display_name"].iloc[0])
        raw_row = comparison.loc[comparison["view_code"] == "raw"].iloc[0]
        centered_row = comparison.loc[comparison["view_code"] == "row_column_centered"].iloc[0]
        lines.extend(
            [
                f"### {display_name}",
                "",
                f"- Raw matrix: `{raw_row['low_rank_strength']}` low-rank evidence; top component explains `{format_float(float(raw_row['leading_energy_share']))}` of spectral energy and `{int(raw_row['effective_rank_95'])}` components explain `95%`.",
                f"- After row-and-column centering: `{centered_row['low_rank_strength']}` low-rank evidence; top component explains `{format_float(float(centered_row['leading_energy_share']))}` of spectral energy and `{int(centered_row['effective_rank_95'])}` components explain `95%`.",
                f"- Elbow summary after row-and-column centering: strongest adjacent singular-value drop at rank `{int(centered_row['elbow_rank'])}` with ratio `{format_float(float(centered_row['elbow_ratio']))}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            "",
            "- `continuous_death_spectrum.csv`",
            "- `binary_death_spectrum.csv`",
            "- `continuous_death_centering_comparison.csv`",
            "- `binary_death_centering_comparison.csv`",
            "- `continuous_death_scree.png`",
            "- `continuous_death_cumulative_energy.png`",
            "- `binary_death_scree.png`",
            "- `binary_death_cumulative_energy.png`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff_date = pd.Timestamp(args.cutoff_date)
    weekly_panel = load_panel(WEEKLY_PANEL_PATH)
    binary_panel = load_panel(BINARY_PANEL_PATH)
    if args.node_index_path is not None:
        allowed_fips = load_allowed_fips(Path(args.node_index_path))
        weekly_panel = restrict_panel_to_fips(weekly_panel, allowed_fips)
        binary_panel = restrict_panel_to_fips(binary_panel, allowed_fips)

    first_positive_date = first_positive_vaccination_date(weekly_panel)
    first_nonnull_series = weekly_panel.loc[
        weekly_panel[VACCINATION_COLUMN].notna(),
        "WeekEndDate",
    ]
    if first_nonnull_series.empty:
        raise ValueError("No non-missing vaccination coverage was found.")
    first_nonnull_date = pd.Timestamp(first_nonnull_series.min())

    continuous_bundle = build_matrix_bundle(
        weekly_panel,
        matrix_code="continuous_death",
        display_name="Continuous death-rate matrix",
        value_column=CONTINUOUS_COLUMN,
        cutoff_date=cutoff_date,
    )
    binary_bundle = build_matrix_bundle(
        binary_panel,
        matrix_code="binary_death",
        display_name="Binary death-threshold matrix",
        value_column=BINARY_COLUMN,
        cutoff_date=cutoff_date,
    )

    bundles = [continuous_bundle, binary_bundle]
    comparison_frames: list[pd.DataFrame] = []

    for bundle in bundles:
        spectra: list[pd.DataFrame] = []
        summaries: list[dict[str, object]] = []
        for view_code in VIEW_LABELS:
            transformed = apply_view(bundle.matrix, view_code)
            spectrum, summary = summarize_view(
                bundle.matrix_code,
                bundle.display_name,
                transformed,
                view_code,
            )
            spectra.append(spectrum)
            summaries.append(summary)

        spectrum_frame = pd.concat(spectra, ignore_index=True)
        comparison_frame = pd.DataFrame(summaries)
        comparison_frames.append(comparison_frame)

        spectrum_frame.to_csv(
            output_dir / f"{bundle.matrix_code}_spectrum.csv",
            index=False,
        )
        comparison_frame.to_csv(
            output_dir / f"{bundle.matrix_code}_centering_comparison.csv",
            index=False,
        )
        render_scree_plot(
            spectrum_frame,
            output_dir / f"{bundle.matrix_code}_scree.png",
        )
        render_cumulative_plot(
            spectrum_frame,
            output_dir / f"{bundle.matrix_code}_cumulative_energy.png",
        )

    summary_text = build_summary_markdown(
        cutoff_date=cutoff_date,
        first_positive_date=first_positive_date,
        first_nonnull_date=first_nonnull_date,
        scope_label=str(args.scope_label),
        bundles=bundles,
        comparison_frames=comparison_frames,
    )
    (output_dir / "summary.md").write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
