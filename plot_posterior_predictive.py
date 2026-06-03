"""Generate figures from posterior predictive and counterfactual summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_time_level_pp_comparison(
    observed_df: pd.DataFrame,
    pp_main_df: pd.DataFrame,
    pp_xi_zero_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot time-level posterior predictive comparison with three lines."""
    fig, ax = plt.subplots(figsize=(16, 8))

    x = observed_df["time_index"].values

    ax.plot(x, observed_df["observed_value"].values, label="Observed", linewidth=3, color="black", zorder=4)

    ax.fill_between(
        x,
        pp_main_df["sample_mean"].values - pp_main_df["sample_std"].values,
        pp_main_df["sample_mean"].values + pp_main_df["sample_std"].values,
        alpha=0.2, color="coral",
    )
    ax.plot(x, pp_main_df["sample_mean"].values, label="Estimated", linewidth=3, color="coral", zorder=3)

    ax.fill_between(
        x,
        pp_xi_zero_df["sample_mean"].values - pp_xi_zero_df["sample_std"].values,
        pp_xi_zero_df["sample_mean"].values + pp_xi_zero_df["sample_std"].values,
        alpha=0.2, color="green",
    )
    ax.plot(x, pp_xi_zero_df["sample_mean"].values, label="Estimated (No Interference)", linewidth=3, color="green", zorder=2)

    ax.axvline(50, color="gray", linestyle="--", linewidth=2, alpha=0.7)

    ax.set_xlabel("Time Index", fontsize=28)
    ax.set_ylabel("Outcome Value", fontsize=28)
    ax.set_title("Estimate vs Observed Average Outcome over Time", fontsize=32)
    ax.legend(fontsize=20, loc="best")
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_unit_level_pp_comparison(
    observed_df: pd.DataFrame,
    pp_main_df: pd.DataFrame,
    pp_xi_zero_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot unit-level posterior predictive comparison with three lines."""
    fig, ax = plt.subplots(figsize=(16, 10))

    sort_idx = np.argsort(observed_df["observed_value"].values)
    x = np.arange(len(sort_idx))

    observed = observed_df["observed_value"].values[sort_idx]
    pp_main_mean = pp_main_df["sample_mean"].values[sort_idx]
    pp_main_std = pp_main_df["sample_std"].values[sort_idx]
    pp_xi_zero_mean = pp_xi_zero_df["sample_mean"].values[sort_idx]
    pp_xi_zero_std = pp_xi_zero_df["sample_std"].values[sort_idx]

    ax.plot(x, observed, label="Observed", linewidth=3, color="black", zorder=4)
    ax.fill_between(x, pp_main_mean - pp_main_std, pp_main_mean + pp_main_std, alpha=0.2, color="coral")
    ax.plot(x, pp_main_mean, label="Estimated", linewidth=3, color="coral", zorder=3)
    ax.fill_between(x, pp_xi_zero_mean - pp_xi_zero_std, pp_xi_zero_mean + pp_xi_zero_std, alpha=0.2, color="green")
    ax.plot(x, pp_xi_zero_mean, label="Estimated (No Interference)", linewidth=3, color="green", zorder=2)

    ax.set_xlabel("Unit Index (sorted by observed value)", fontsize=28)
    ax.set_ylabel("Outcome Value", fontsize=28)
    ax.set_title("Estimated vs Observed Average Outcome per County", fontsize=32)
    ax.legend(fontsize=20, loc="best")
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_time_level_counterfactual_with_interference(
    observed_df: pd.DataFrame,
    cf_no_int_df: pd.DataFrame,
    cf_all_int_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot time-level counterfactual comparison with interference."""
    fig, ax = plt.subplots(figsize=(16, 8))

    x = observed_df["time_index"].values

    ax.plot(x, observed_df["observed_value"].values, label="Observed", linewidth=3, color="black", zorder=3)

    ax.fill_between(x, cf_no_int_df["sample_mean"].values - cf_no_int_df["sample_std"].values,
                    cf_no_int_df["sample_mean"].values + cf_no_int_df["sample_std"].values, alpha=0.15, color="steelblue")
    ax.plot(x, cf_no_int_df["sample_mean"].values, label="No Interventions", linewidth=3, color="steelblue", zorder=2)

    ax.fill_between(x, cf_all_int_df["sample_mean"].values - cf_all_int_df["sample_std"].values,
                    cf_all_int_df["sample_mean"].values + cf_all_int_df["sample_std"].values, alpha=0.15, color="coral")
    ax.plot(x, cf_all_int_df["sample_mean"].values, label="All Interventions", linewidth=3, color="coral", zorder=1)

    ax.axvline(50, color="gray", linestyle="--", linewidth=2, alpha=0.7)

    ax.set_xlabel("Time Index", fontsize=28)
    ax.set_ylabel("Outcome Value", fontsize=28)
    ax.set_title("Counterfactual Scenarios", fontsize=32)
    ax.legend(fontsize=20, loc="best")
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_time_level_counterfactual_no_interference(
    observed_df: pd.DataFrame,
    cf_no_int_xi_zero_df: pd.DataFrame,
    cf_all_int_xi_zero_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot time-level counterfactual comparison without interference."""
    fig, ax = plt.subplots(figsize=(16, 8))

    x = observed_df["time_index"].values

    ax.plot(x, observed_df["observed_value"].values, label="Observed", linewidth=3, color="black", zorder=3)

    ax.fill_between(x, cf_no_int_xi_zero_df["sample_mean"].values - cf_no_int_xi_zero_df["sample_std"].values,
                    cf_no_int_xi_zero_df["sample_mean"].values + cf_no_int_xi_zero_df["sample_std"].values, alpha=0.15, color="steelblue")
    ax.plot(x, cf_no_int_xi_zero_df["sample_mean"].values, label="No Interventions", linewidth=3, color="steelblue", zorder=2)

    ax.fill_between(x, cf_all_int_xi_zero_df["sample_mean"].values - cf_all_int_xi_zero_df["sample_std"].values,
                    cf_all_int_xi_zero_df["sample_mean"].values + cf_all_int_xi_zero_df["sample_std"].values, alpha=0.15, color="coral")
    ax.plot(x, cf_all_int_xi_zero_df["sample_mean"].values, label="All Interventions", linewidth=3, color="coral", zorder=1)

    ax.axvline(50, color="gray", linestyle="--", linewidth=2, alpha=0.7)

    ax.set_xlabel("Time Index", fontsize=28)
    ax.set_ylabel("Outcome Value", fontsize=28)
    ax.set_title("Counterfactual Scenarios (No Interference)", fontsize=32)
    ax.legend(fontsize=20, loc="best")
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_unit_level_counterfactual_with_interference(
    observed_df: pd.DataFrame,
    cf_no_int_df: pd.DataFrame,
    cf_all_int_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot unit-level counterfactual comparison with interference."""
    fig, ax = plt.subplots(figsize=(16, 10))

    sort_idx = np.argsort(observed_df["observed_value"].values)
    x = np.arange(len(sort_idx))

    observed = observed_df["observed_value"].values[sort_idx]
    cf_no_int_mean = cf_no_int_df["sample_mean"].values[sort_idx]
    cf_no_int_std = cf_no_int_df["sample_std"].values[sort_idx]
    cf_all_int_mean = cf_all_int_df["sample_mean"].values[sort_idx]
    cf_all_int_std = cf_all_int_df["sample_std"].values[sort_idx]

    ax.plot(x, observed, label="Observed", linewidth=3, color="black", zorder=3)
    ax.fill_between(x, cf_no_int_mean - cf_no_int_std, cf_no_int_mean + cf_no_int_std, alpha=0.15, color="steelblue")
    ax.plot(x, cf_no_int_mean, label="No Interventions", linewidth=3, color="steelblue", zorder=2)
    ax.fill_between(x, cf_all_int_mean - cf_all_int_std, cf_all_int_mean + cf_all_int_std, alpha=0.15, color="coral")
    ax.plot(x, cf_all_int_mean, label="All Interventions", linewidth=3, color="coral", zorder=1)

    ax.set_xlabel("Unit Index (sorted by observed value)", fontsize=28)
    ax.set_ylabel("Outcome Value", fontsize=28)
    ax.set_title("Counterfactual Scenarios per County", fontsize=32)
    ax.legend(fontsize=20, loc="best")
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_unit_level_counterfactual_no_interference(
    observed_df: pd.DataFrame,
    cf_no_int_xi_zero_df: pd.DataFrame,
    cf_all_int_xi_zero_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot unit-level counterfactual comparison without interference."""
    fig, ax = plt.subplots(figsize=(16, 10))

    sort_idx = np.argsort(observed_df["observed_value"].values)
    x = np.arange(len(sort_idx))

    observed = observed_df["observed_value"].values[sort_idx]
    cf_no_int_xi_zero_mean = cf_no_int_xi_zero_df["sample_mean"].values[sort_idx]
    cf_no_int_xi_zero_std = cf_no_int_xi_zero_df["sample_std"].values[sort_idx]
    cf_all_int_xi_zero_mean = cf_all_int_xi_zero_df["sample_mean"].values[sort_idx]
    cf_all_int_xi_zero_std = cf_all_int_xi_zero_df["sample_std"].values[sort_idx]

    ax.plot(x, observed, label="Observed", linewidth=3, color="black", zorder=3)
    ax.fill_between(x, cf_no_int_xi_zero_mean - cf_no_int_xi_zero_std, cf_no_int_xi_zero_mean + cf_no_int_xi_zero_std, alpha=0.15, color="steelblue")
    ax.plot(x, cf_no_int_xi_zero_mean, label="No Interventions", linewidth=3, color="steelblue", zorder=2)
    ax.fill_between(x, cf_all_int_xi_zero_mean - cf_all_int_xi_zero_std, cf_all_int_xi_zero_mean + cf_all_int_xi_zero_std, alpha=0.15, color="coral")
    ax.plot(x, cf_all_int_xi_zero_mean, label="All Interventions", linewidth=3, color="coral", zorder=1)

    ax.set_xlabel("Unit Index (sorted by observed value)", fontsize=28)
    ax.set_ylabel("Outcome Value", fontsize=28)
    ax.set_title("Counterfactual Scenarios per County (No Interference)", fontsize=32)
    ax.legend(fontsize=20, loc="best")
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate posterior predictive and counterfactual visualizations.")
    parser.add_argument("--observed_time", type=Path)
    parser.add_argument("--observed_unit", type=Path)
    parser.add_argument("--pp_main_time", type=Path)
    parser.add_argument("--pp_main_unit", type=Path)
    parser.add_argument("--pp_xi_zero_time", type=Path)
    parser.add_argument("--pp_xi_zero_unit", type=Path)
    parser.add_argument("--cf_no_int_time", type=Path)
    parser.add_argument("--cf_no_int_unit", type=Path)
    parser.add_argument("--cf_all_int_time", type=Path)
    parser.add_argument("--cf_all_int_unit", type=Path)
    parser.add_argument("--cf_no_int_xi_zero_time", type=Path)
    parser.add_argument("--cf_no_int_xi_zero_unit", type=Path)
    parser.add_argument("--cf_all_int_xi_zero_time", type=Path)
    parser.add_argument("--cf_all_int_xi_zero_unit", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pp_main_time and args.pp_main_unit and args.pp_xi_zero_time and args.pp_xi_zero_unit:
        print("Loading posterior predictive data...")
        observed_time = pd.read_csv(args.observed_time)
        pp_main_time = pd.read_csv(args.pp_main_time)
        pp_xi_zero_time = pd.read_csv(args.pp_xi_zero_time)
        observed_unit = pd.read_csv(args.observed_unit)
        pp_main_unit = pd.read_csv(args.pp_main_unit)
        pp_xi_zero_unit = pd.read_csv(args.pp_xi_zero_unit)

        plot_time_level_pp_comparison(observed_time, pp_main_time, pp_xi_zero_time, args.output_dir / "pp_time_comparison.png")
        plot_unit_level_pp_comparison(observed_unit, pp_main_unit, pp_xi_zero_unit, args.output_dir / "pp_unit_comparison.png")

    if (args.cf_no_int_time and args.cf_no_int_unit and args.cf_all_int_time and args.cf_all_int_unit
            and args.cf_no_int_xi_zero_time and args.cf_no_int_xi_zero_unit
            and args.cf_all_int_xi_zero_time and args.cf_all_int_xi_zero_unit):
        print("Loading counterfactual data...")
        observed_time = pd.read_csv(args.observed_time)
        observed_unit = pd.read_csv(args.observed_unit)
        cf_no_int_time = pd.read_csv(args.cf_no_int_time)
        cf_no_int_xi_zero_time = pd.read_csv(args.cf_no_int_xi_zero_time)
        cf_all_int_time = pd.read_csv(args.cf_all_int_time)
        cf_all_int_xi_zero_time = pd.read_csv(args.cf_all_int_xi_zero_time)
        cf_no_int_unit = pd.read_csv(args.cf_no_int_unit)
        cf_no_int_xi_zero_unit = pd.read_csv(args.cf_no_int_xi_zero_unit)
        cf_all_int_unit = pd.read_csv(args.cf_all_int_unit)
        cf_all_int_xi_zero_unit = pd.read_csv(args.cf_all_int_xi_zero_unit)

        plot_time_level_counterfactual_with_interference(observed_time, cf_no_int_time, cf_all_int_time, args.output_dir / "cf_time_with_interference.png")
        plot_time_level_counterfactual_no_interference(observed_time, cf_no_int_xi_zero_time, cf_all_int_xi_zero_time, args.output_dir / "cf_time_no_interference.png")
        plot_unit_level_counterfactual_with_interference(observed_unit, cf_no_int_unit, cf_all_int_unit, args.output_dir / "cf_unit_with_interference.png")
        plot_unit_level_counterfactual_no_interference(observed_unit, cf_no_int_xi_zero_unit, cf_all_int_xi_zero_unit, args.output_dir / "cf_unit_no_interference.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
