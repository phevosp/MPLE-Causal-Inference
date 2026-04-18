"""Small regression tests for the minimal latent-only MPLE pipeline."""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.synthetic_data_generation import load_fixed_intervention_artifacts
from mple import _canonicalize_theta
from model_utils import (
    ModelArtifacts,
    build_fit_model_artifacts,
    build_synthetic_field,
    compose_interaction_matrix,
    compose_latent_field_matrix,
    get_xi,
    interaction_effect,
    interaction_matrix_infinity_norm,
    load_model_artifacts,
    load_true_parameters,
    parameter_names,
    save_model_artifacts,
    unpack_theta,
)
from posterior_predictive_utils import (
    compute_panel_statistics,
    load_fit_parameter_bundle,
    load_truth_parameter_bundle,
    summarize_predictive_statistics,
)
from report_posterior_predictive import (
    collect_predictive_rows,
    group_and_rank_predictive_rows,
)
from report_parameter_recovery_detailed import (
    collect_fit_rows,
    group_and_rank_fit_rows,
    write_fit_reports,
)
from run_fit_pipeline import run_fits
from run_generation_pipeline import run_generation
from run_posterior_predictive_pipeline import (
    _index_generation_rows,
    _resolve_fit_lookup,
    _resolve_target_pairs,
    run_posterior_predictive,
)


def base_config() -> object:
    return OmegaConf.create(
        {
            "global_params": {
                "N": 4,
                "T": 3,
                "B": 1.0,
                "latent_rank": 0,
            },
            "estimation_params": {
                "xi": 0.25,
                "beta": 0.1,
                "eta": 0.2,
                "zeta": -0.1,
                "psi": 0.3,
            },
            "generation_params": {
                "seed": 0,
                "intervention_mode": "generated_z",
            },
        }
    )


class MinimalPipelineTests(unittest.TestCase):
    def test_rank_zero_field_realizes_zero_external_field(self) -> None:
        config = base_config()
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        artifacts = build_synthetic_field(config, gamma)
        self.assertEqual(artifacts.latent_rank, 0)
        self.assertEqual(artifacts.node_factors.shape, (4, 0))
        self.assertEqual(artifacts.time_factors.shape, (3, 0))
        self.assertTrue(np.allclose(artifacts.field_matrix, 0.0))

    def test_xi_is_scalar(self) -> None:
        config = base_config()
        self.assertEqual(get_xi(config), 0.25)
        with self.assertRaises(ValueError):
            compose_interaction_matrix(np.array([0.1, 0.2]), np.eye(4))

    def test_interaction_feature_shape_is_2d(self) -> None:
        x = np.array([[1, -1, 1, -1], [-1, -1, 1, 1]], dtype=float)
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        features = interaction_effect(x, gamma)
        self.assertEqual(features.shape, (2, 4))

    def test_fixed_z_loader_checks_shape(self) -> None:
        config = base_config()
        config.generation_params.intervention_mode = "fixed_z"
        root = REPO_ROOT / "experiments" / f".tmp_fixed_z_test_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            np.savez(root / "panel_data.npz", x=np.ones((3, 4)), z=-np.ones((3, 4)))
            np.save(root / "z_0.npy", -np.ones(4))
            config.generation_params.fixed_z_source = {
                "panel_path": str(root / "panel_data.npz"),
                "z0_path": str(root / "z_0.npy"),
            }
            z, z0, metadata = load_fixed_intervention_artifacts(config)
            self.assertEqual(z.shape, (3, 4))
            self.assertEqual(z0.shape, (4,))
            self.assertIn("fixed_z_panel_path", metadata)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_positive_rank_latent_field_is_realized(self) -> None:
        config = base_config()
        config.global_params.latent_rank = 2
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        artifacts = build_synthetic_field(config, gamma)
        field_matrix = compose_latent_field_matrix(
            artifacts.node_factors, artifacts.time_factors
        )
        self.assertEqual(artifacts.latent_rank, 2)
        self.assertEqual(artifacts.node_factors.shape, (4, 2))
        self.assertEqual(artifacts.time_factors.shape, (3, 2))
        self.assertEqual(field_matrix.shape, (3, 4))
        self.assertLessEqual(
            float(np.linalg.norm(field_matrix, ord=np.inf)), 1.0 + 1e-8
        )

    def test_model_artifact_roundtrip_and_true_parameter_loading(self) -> None:
        config = base_config()
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        artifacts = build_synthetic_field(config, gamma)
        root = REPO_ROOT / "experiments" / f".tmp_model_artifacts_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            save_model_artifacts(root, artifacts)
            loaded = load_model_artifacts(root)
            theta = load_true_parameters(config, loaded)
            self.assertEqual(loaded.t_steps, 3)
            self.assertEqual(theta.shape[0], 5)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_canonicalization_enforces_B_on_scalars_and_interaction(self) -> None:
        gamma = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=1,
            latent_rank=0,
        )
        theta = np.array([5.0, 4.0, -3.5, 2.5, -2.2], dtype=float)
        projected = _canonicalize_theta(
            theta=theta,
            artifacts=artifacts,
            fit_intervention_model=True,
            bound_B=1.0,
        )
        parts = unpack_theta(projected, artifacts, fit_intervention_model=True)
        self.assertLessEqual(abs(float(parts["beta"])), 1.0 + 1e-12)
        self.assertLessEqual(abs(float(parts["eta"])), 1.0 + 1e-12)
        self.assertLessEqual(abs(float(parts["zeta"])), 1.0 + 1e-12)
        self.assertLessEqual(abs(float(parts["psi"])), 1.0 + 1e-12)
        interaction = compose_interaction_matrix(float(parts["xi"]), artifacts.gamma_matrix)
        self.assertLessEqual(
            interaction_matrix_infinity_norm(interaction),
            1.0 + 1e-12,
        )

    def test_parameter_names_and_unpack_respect_fixed_scalars(self) -> None:
        artifacts = ModelArtifacts(
            gamma_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            t_steps=1,
            latent_rank=0,
        )
        theta = np.array([0.1, 0.25, 0.4], dtype=float)
        names = parameter_names(
            artifacts,
            fit_intervention_model=True,
            fixed_scalar_params={"beta": 0.0, "psi": 0.3},
        )
        self.assertEqual(names, ["xi", "eta", "zeta"])
        parts = unpack_theta(
            theta,
            artifacts,
            fit_intervention_model=True,
            fixed_scalar_params={"beta": 0.0, "psi": 0.3},
        )
        self.assertEqual(parts["beta"], 0.0)
        self.assertEqual(parts["psi"], 0.3)
        self.assertAlmostEqual(parts["xi"], 0.1)
        self.assertAlmostEqual(parts["eta"], 0.25)
        self.assertAlmostEqual(parts["zeta"], 0.4)
        self.assertEqual(parts["node_factors"].shape, (2, 0))
        self.assertEqual(parts["time_factors"].shape, (1, 0))

    def test_build_fit_model_artifacts_uses_latent_rank_only(self) -> None:
        config = base_config()
        config.global_params.N = 5
        config.global_params.T = 7
        config.global_params.latent_rank = 3
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        artifacts = build_fit_model_artifacts(config, gamma)
        self.assertEqual(artifacts.latent_rank, 3)
        self.assertEqual(artifacts.t_steps, 7)
        self.assertIsNone(artifacts.field_matrix)


class FitReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "experiments" / f".tmp_fit_reporting_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_fit(
        self,
        experiment_name: str,
        variant_name: str,
        summary_entries: dict[str, tuple[float | None, float | None, float | None]],
        *,
        descriptor: str = "",
        intervention_source: str = "generated",
        graph_source: str = "generated",
        latent_rank: int = 0,
        B: float = 1.0,
        fit_intervention_model: bool = True,
        fixed_scalar_params: str = "{}",
    ) -> dict[str, object]:
        experiment_root = self.root / experiment_name
        fit_root = experiment_root / "fits" / variant_name
        fit_root.mkdir(parents=True, exist_ok=True)

        summary_path = fit_root / "mple_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["category", "name", "estimate", "true", "squared_error"],
            )
            writer.writeheader()
            for name, (estimate, truth, squared_error) in summary_entries.items():
                writer.writerow(
                    {
                        "category": "metric" if name in {"final_loss", "field_rmse", "interaction_fro_error"} else "scalar",
                        "name": name,
                        "estimate": "" if estimate is None else estimate,
                        "true": "" if truth is None else truth,
                        "squared_error": "" if squared_error is None else squared_error,
                    }
                )

        (fit_root / "mple.log").write_text(
            "2026-01-01 00:00:00 | INFO | Optimizer status: CONVERGED\n",
            encoding="utf-8",
        )
        OmegaConf.save(
            OmegaConf.create(
                {
                    "variant_name": variant_name,
                    "experiment_name": experiment_name,
                    "latent_rank": latent_rank,
                }
            ),
            fit_root / "fit_metadata.yaml",
        )
        OmegaConf.save(
            OmegaConf.create(
                {
                    "global_params": {"B": B, "latent_rank": latent_rank},
                    "estimation_params": {
                        "fit_intervention_model": fit_intervention_model,
                        "fixed_scalar_params": {},
                    },
                }
            ),
            fit_root / "fit_realized_config.yaml",
        )
        return {
            "experiment_name": experiment_name,
            "experiment_slug": experiment_name,
            "descriptor": descriptor or experiment_name,
            "experiment_path": str(experiment_root.resolve()),
            "intervention_source": intervention_source,
            "graph_source": graph_source,
            "variant_name": variant_name,
            "variant_slug": variant_name,
            "fit_path": str(fit_root.resolve()),
            "N": 5,
            "T": 4,
            "s": 1,
            "B": B,
            "latent_rank": latent_rank,
            "fit_intervention_model": fit_intervention_model,
            "fixed_scalar_params": fixed_scalar_params,
            "status": "completed",
        }

    def _write_manifest(self, rows: list[dict[str, object]]) -> Path:
        manifest_path = self.root / "fit_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return manifest_path

    def test_reporter_selects_best_variant_by_total_recovery_rmse(self) -> None:
        manifest_path = self._write_manifest(
            [
                self._write_fit(
                    "exp_a",
                    "variant_low_field_high_scalar",
                    {
                        "final_loss": (0.40, None, None),
                        "field_rmse": (0.10, None, None),
                        "interaction_fro_error": (0.20, None, None),
                        "beta": (1.00, 0.00, 1.00),
                    },
                ),
                self._write_fit(
                    "exp_a",
                    "variant_better_total",
                    {
                        "final_loss": (0.60, None, None),
                        "field_rmse": (0.20, None, None),
                        "interaction_fro_error": (0.50, None, None),
                        "beta": (0.05, 0.00, 0.0025),
                    },
                ),
            ]
        )
        rows = collect_fit_rows(manifest_path)
        grouped, winners = group_and_rank_fit_rows(rows)
        ranked = grouped[str((self.root / "exp_a").resolve())]
        self.assertEqual(ranked[0]["ranking_mode"], "total_recovery_rmse")
        self.assertAlmostEqual(ranked[0]["total_recovery_rmse"], 0.25)
        self.assertEqual(ranked[0]["variant_name"], "variant_better_total")
        self.assertTrue(ranked[0]["is_best"])
        self.assertEqual(winners[0]["variant_name"], "variant_better_total")

    def test_reporter_uses_tie_breakers(self) -> None:
        manifest_path = self._write_manifest(
            [
                self._write_fit(
                    "exp_tie",
                    "variant_interaction",
                    {
                        "final_loss": (0.40, None, None),
                        "field_rmse": (0.10, None, None),
                        "interaction_fro_error": (0.30, None, None),
                    },
                ),
                self._write_fit(
                    "exp_tie",
                    "variant_loss",
                    {
                        "final_loss": (0.20, None, None),
                        "field_rmse": (0.10, None, None),
                        "interaction_fro_error": (0.10, None, None),
                    },
                ),
            ]
        )
        rows = collect_fit_rows(manifest_path)
        grouped, _ = group_and_rank_fit_rows(rows)
        ranked = grouped[str((self.root / "exp_tie").resolve())]
        self.assertEqual(ranked[0]["variant_name"], "variant_loss")
        self.assertEqual(ranked[0]["rank_in_experiment"], 1)
        self.assertEqual(ranked[1]["rank_in_experiment"], 2)

    def test_reporter_falls_back_to_final_loss_without_truth_metrics(self) -> None:
        manifest_path = self._write_manifest(
            [
                self._write_fit(
                    "exp_notruth",
                    "variant_slow",
                    {"final_loss": (0.50, None, None)},
                ),
                self._write_fit(
                    "exp_notruth",
                    "variant_fast",
                    {"final_loss": (0.20, None, None)},
                ),
            ]
        )
        rows = collect_fit_rows(manifest_path)
        grouped, winners = group_and_rank_fit_rows(rows)
        ranked = grouped[str((self.root / "exp_notruth").resolve())]
        self.assertEqual(ranked[0]["ranking_mode"], "final_loss_only")
        self.assertEqual(ranked[0]["variant_name"], "variant_fast")
        self.assertEqual(winners[0]["variant_name"], "variant_fast")

    def test_run_fit_pipeline_writes_grouped_reports(self) -> None:
        generation_spec_path = self.root / "generation_spec.yaml"
        fits_spec_path = self.root / "fits_spec.yaml"
        generation_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    f"  experiment_root: {self.root.as_posix()}/generated",
                    f"  manifest_path: {self.root.as_posix()}/generated/generation_manifest.csv",
                    "  dimensions:",
                    "    N: 6",
                    "    T: 4",
                    "    s: 1",
                    "  generation:",
                    "    gibbs_sweeps: 1",
                    "    seed: 7",
                    "  x0:",
                    "    generator: bernoulli",
                    "    params:",
                    "      p: 0.5",
                    "      fixed_val: null",
                    "  graph:",
                    "    source: generated",
                    "    generator: erdos_renyi",
                    "    params:",
                    "      p: 0.5",
                    "    artifact:",
                    "      gamma_path: null",
                    "      node_index_path: null",
                    "      artifact_dir: null",
                    "      network_name: null",
                    "      trim_scope: null",
                    "  intervention:",
                    "    source: generated",
                    "    artifact:",
                    "      panel_path: null",
                    "      z0_path: null",
                    "      artifact_dir: null",
                    "      shared_panel_dir: null",
                    "      outcome_code: null",
                    "      intervention_code: null",
                    "      lag_code: null",
                    "      trim_scope: null",
                    "  truth:",
                    "    B: 1.0",
                    "    latent_rank: 0",
                    "    scalars:",
                    "      beta: 0.2",
                    "      xi: 0.1",
                    "      eta: 0.05",
                    "      zeta: -0.1",
                    "      psi: 0.2",
                    "experiments:",
                    "  - name: smoke_rank_0",
                ]
            ),
            encoding="utf-8",
        )
        fits_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    "  fit_root_name: fits",
                    f"  fit_manifest_path: {self.root.as_posix()}/generated/fit_manifest.csv",
                    "  optimizer:",
                    "    steps: 5",
                    "    tol: 1.0e-6",
                    "    seed: 0",
                    "  B: 1.0",
                    "  latent_rank: 0",
                    "  estimation:",
                    "    fit_intervention_model: true",
                    "    beta_mask_pre_intervention: false",
                    "    beta_mask_rescale: false",
                    "    fixed_scalar_params: {}",
                    "variants:",
                    "  - name: rank_0",
                    "  - name: rank_0_fixed_xi",
                    "    estimation:",
                    "      fixed_scalar_params:",
                    "        xi: 0.0",
                ]
            ),
            encoding="utf-8",
        )

        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        fit_manifest = run_fits(generation_manifest, fits_spec_path, overwrite=True)

        experiment_root = self.root / "generated" / "smoke_rank_0"
        fit_summary_csv = experiment_root / "fit_summary.csv"
        fit_summary_md = experiment_root / "fit_summary.md"
        winners_csv = self.root / "generated" / "best_fit_by_experiment.csv"
        winners_md = self.root / "generated" / "best_fit_by_experiment.md"

        self.assertTrue(fit_summary_csv.exists())
        self.assertTrue(fit_summary_md.exists())
        self.assertTrue(winners_csv.exists())
        self.assertTrue(winners_md.exists())

        with fit_summary_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(sum(row["is_best"] == "True" for row in rows), 1)
        self.assertEqual(len(rows), 2)
        self.assertIn("total_recovery_rmse", rows[0])

        with winners_csv.open("r", encoding="utf-8", newline="") as handle:
            winner_rows = list(csv.DictReader(handle))
        self.assertEqual(len(winner_rows), 1)
        self.assertEqual(winner_rows[0]["experiment_name"], "smoke_rank_0")
        self.assertIn("total_recovery_rmse", winner_rows[0])
        self.assertEqual(Path(fit_manifest), self.root / "generated" / "fit_manifest.csv")


class PosteriorPredictiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "experiments" / f".tmp_predictive_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_manifest(self, rows: list[dict[str, object]]) -> Path:
        manifest_path = self.root / "posterior_predictive_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return manifest_path

    def _write_target_pairs(self, rows: list[dict[str, object]]) -> Path:
        target_pairs_path = self.root / "target_pairs.csv"
        with target_pairs_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["experiment_name", "source_type", "variant_name"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return target_pairs_path

    def test_compute_panel_statistics_is_hand_checkable(self) -> None:
        x = np.ones((3, 2), dtype=float)
        z = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.ones(2, dtype=float)
        field = np.ones((3, 2), dtype=float)
        gamma = np.zeros((2, 2), dtype=float)

        stats = compute_panel_statistics(
            x,
            z=z,
            x_0=x_0,
            s=1,
            field_matrix=field,
            gamma_matrix=gamma,
        )

        self.assertAlmostEqual(stats["overall_mean_magnetization"], 1.0)
        self.assertAlmostEqual(stats["post_intervention_mean_magnetization"], 1.0)
        self.assertAlmostEqual(stats["lag1_persistence"], 1.0)
        self.assertAlmostEqual(stats["graph_interaction_energy"], 0.0)
        self.assertAlmostEqual(stats["field_alignment"], 1.0)

    def test_predictive_summary_prefers_lower_mean_abs_zscore(self) -> None:
        manifest_path = self._write_manifest(
            [
                {
                    "experiment_name": "exp_a",
                    "experiment_slug": "exp_a",
                    "descriptor": "exp_a",
                    "experiment_path": str((self.root / "exp_a").resolve()),
                    "intervention_source": "generated",
                    "graph_source": "generated",
                    "N": 5,
                    "T": 4,
                    "s": 1,
                    "run_name": "run_one",
                    "run_slug": "run_one",
                    "source_type": "truth",
                    "source_name": "truth",
                    "source_slug": "truth",
                    "latent_rank": 0,
                    "B": 1.0,
                    "fit_intervention_model": True,
                    "num_samples": 8,
                    "gibbs_sweeps": 2,
                    "seed": 0,
                    "mean_abs_zscore": 0.25,
                    "max_abs_zscore": 0.50,
                    "coverage_rate": 1.0,
                    "num_statistics": 10,
                    "output_path": str((self.root / "exp_a" / "posterior_predictive" / "truth" / "run_one").resolve()),
                },
                {
                    "experiment_name": "exp_a",
                    "experiment_slug": "exp_a",
                    "descriptor": "exp_a",
                    "experiment_path": str((self.root / "exp_a").resolve()),
                    "intervention_source": "generated",
                    "graph_source": "generated",
                    "N": 5,
                    "T": 4,
                    "s": 1,
                    "run_name": "run_one",
                    "run_slug": "run_one",
                    "source_type": "fit",
                    "source_name": "rank_0",
                    "source_slug": "fit_rank_0",
                    "latent_rank": 0,
                    "B": 1.0,
                    "fit_intervention_model": True,
                    "num_samples": 8,
                    "gibbs_sweeps": 2,
                    "seed": 0,
                    "mean_abs_zscore": 0.40,
                    "max_abs_zscore": 0.45,
                    "coverage_rate": 0.9,
                    "num_statistics": 10,
                    "output_path": str((self.root / "exp_a" / "posterior_predictive" / "fit_rank_0" / "run_one").resolve()),
                },
            ]
        )
        rows = collect_predictive_rows(manifest_path)
        grouped, winners = group_and_rank_predictive_rows(rows)
        ranked = grouped[str((self.root / "exp_a").resolve())]
        self.assertEqual(ranked[0]["source_name"], "truth")
        self.assertTrue(ranked[0]["is_best"])
        self.assertEqual(winners[0]["source_name"], "truth")

    def test_predictive_summary_uses_max_zscore_tiebreaker(self) -> None:
        manifest_path = self._write_manifest(
            [
                {
                    "experiment_name": "exp_tie",
                    "experiment_slug": "exp_tie",
                    "descriptor": "exp_tie",
                    "experiment_path": str((self.root / "exp_tie").resolve()),
                    "intervention_source": "generated",
                    "graph_source": "generated",
                    "N": 5,
                    "T": 4,
                    "s": 1,
                    "run_name": "run_one",
                    "run_slug": "run_one",
                    "source_type": "fit",
                    "source_name": "variant_worse_max",
                    "source_slug": "fit_variant_worse_max",
                    "latent_rank": 0,
                    "B": 1.0,
                    "fit_intervention_model": True,
                    "num_samples": 8,
                    "gibbs_sweeps": 2,
                    "seed": 0,
                    "mean_abs_zscore": 0.30,
                    "max_abs_zscore": 0.60,
                    "coverage_rate": 0.9,
                    "num_statistics": 10,
                    "output_path": str((self.root / "exp_tie" / "posterior_predictive" / "fit_variant_worse_max" / "run_one").resolve()),
                },
                {
                    "experiment_name": "exp_tie",
                    "experiment_slug": "exp_tie",
                    "descriptor": "exp_tie",
                    "experiment_path": str((self.root / "exp_tie").resolve()),
                    "intervention_source": "generated",
                    "graph_source": "generated",
                    "N": 5,
                    "T": 4,
                    "s": 1,
                    "run_name": "run_one",
                    "run_slug": "run_one",
                    "source_type": "fit",
                    "source_name": "variant_better_max",
                    "source_slug": "fit_variant_better_max",
                    "latent_rank": 0,
                    "B": 1.0,
                    "fit_intervention_model": True,
                    "num_samples": 8,
                    "gibbs_sweeps": 2,
                    "seed": 0,
                    "mean_abs_zscore": 0.30,
                    "max_abs_zscore": 0.40,
                    "coverage_rate": 0.9,
                    "num_statistics": 10,
                    "output_path": str((self.root / "exp_tie" / "posterior_predictive" / "fit_variant_better_max" / "run_one").resolve()),
                },
            ]
        )
        rows = collect_predictive_rows(manifest_path)
        grouped, _ = group_and_rank_predictive_rows(rows)
        ranked = grouped[str((self.root / "exp_tie").resolve())]
        self.assertEqual(ranked[0]["source_name"], "variant_better_max")
        self.assertEqual(ranked[0]["rank_in_experiment"], 1)

    def test_target_pair_resolution_validates_truth_and_fit_rows(self) -> None:
        generation_manifest_path = self.root / "generation_manifest.csv"
        fit_manifest_path = self.root / "fit_manifest.csv"
        generation_rows = [
            {
                "experiment_name": "exp_a",
                "experiment_path": str((self.root / "generated" / "exp_a").resolve()),
            }
        ]
        fit_rows = [
            {
                "experiment_name": "exp_a",
                "variant_name": "rank_0",
                "variant_slug": "rank_0",
                "fit_path": str(
                    (self.root / "generated" / "exp_a" / "fits" / "rank_0").resolve()
                ),
                "B": "1.0",
                "fit_intervention_model": "True",
            }
        ]
        with generation_manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(generation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(generation_rows)
        with fit_manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fit_rows)

        target_pairs_path = self._write_target_pairs(
            [
                {"experiment_name": "exp_a", "source_type": "truth", "variant_name": ""},
                {"experiment_name": "exp_a", "source_type": "fit", "variant_name": "rank_0"},
            ]
        )
        generation_lookup = _index_generation_rows(generation_manifest_path)
        fit_lookup = _resolve_fit_lookup(fit_manifest_path)
        resolved = _resolve_target_pairs(
            target_pairs_path,
            generation_lookup,
            fit_lookup,
        )
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0]["source_slug"], "truth")
        self.assertEqual(resolved[1]["source_slug"], "fit_rank_0")

    def test_target_pair_resolution_rejects_missing_experiment(self) -> None:
        target_pairs_path = self._write_target_pairs(
            [
                {
                    "experiment_name": "missing_exp",
                    "source_type": "truth",
                    "variant_name": "",
                }
            ]
        )
        with self.assertRaises(ValueError):
            _resolve_target_pairs(target_pairs_path, {}, {})

    def test_target_pair_resolution_rejects_missing_fit_variant(self) -> None:
        target_pairs_path = self._write_target_pairs(
            [
                {
                    "experiment_name": "exp_a",
                    "source_type": "fit",
                    "variant_name": "rank_missing",
                }
            ]
        )
        with self.assertRaises(ValueError):
            _resolve_target_pairs(
                target_pairs_path,
                {"exp_a": {"experiment_name": "exp_a", "experiment_path": "unused"}},
                {},
            )

    def test_summarize_predictive_statistics_returns_scores(self) -> None:
        observed = {"overall_mean_magnetization": 1.0, "field_alignment": 0.5}
        simulated = [
            {"overall_mean_magnetization": 0.8, "field_alignment": 0.3},
            {"overall_mean_magnetization": 0.9, "field_alignment": 0.4},
            {"overall_mean_magnetization": 1.1, "field_alignment": 0.5},
        ]
        rows, summary = summarize_predictive_statistics(observed, simulated)
        self.assertEqual(len(rows), 2)
        self.assertIn("mean_abs_zscore", summary)
        self.assertIn("coverage_rate", summary)
        self.assertGreaterEqual(summary["num_statistics"], 1)

    def test_run_posterior_predictive_pipeline_writes_grouped_reports(self) -> None:
        generation_spec_path = self.root / "generation_spec.yaml"
        fits_spec_path = self.root / "fits_spec.yaml"
        predictive_spec_path = self.root / "posterior_predictive_spec.yaml"
        target_pairs_path = self.root / "target_pairs.csv"
        generation_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    f"  experiment_root: {self.root.as_posix()}/generated",
                    f"  manifest_path: {self.root.as_posix()}/generated/generation_manifest.csv",
                    "  dimensions:",
                    "    N: 6",
                    "    T: 4",
                    "    s: 1",
                    "  generation:",
                    "    gibbs_sweeps: 1",
                    "    seed: 7",
                    "  x0:",
                    "    generator: bernoulli",
                    "    params:",
                    "      p: 0.5",
                    "      fixed_val: null",
                    "  graph:",
                    "    source: generated",
                    "    generator: erdos_renyi",
                    "    params:",
                    "      p: 0.5",
                    "    artifact:",
                    "      gamma_path: null",
                    "      node_index_path: null",
                    "      artifact_dir: null",
                    "      network_name: null",
                    "      trim_scope: null",
                    "  intervention:",
                    "    source: generated",
                    "    artifact:",
                    "      panel_path: null",
                    "      z0_path: null",
                    "      artifact_dir: null",
                    "      shared_panel_dir: null",
                    "      outcome_code: null",
                    "      intervention_code: null",
                    "      lag_code: null",
                    "      trim_scope: null",
                    "  truth:",
                    "    B: 1.0",
                    "    latent_rank: 0",
                    "    scalars:",
                    "      beta: 0.2",
                    "      xi: 0.1",
                    "      eta: 0.05",
                    "      zeta: -0.1",
                    "      psi: 0.2",
                    "experiments:",
                    "  - name: smoke_rank_0",
                ]
            ),
            encoding="utf-8",
        )
        fits_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    "  fit_root_name: fits",
                    f"  fit_manifest_path: {self.root.as_posix()}/generated/fit_manifest.csv",
                    "  optimizer:",
                    "    steps: 5",
                    "    tol: 1.0e-6",
                    "    seed: 0",
                    "  B: 1.0",
                    "  latent_rank: 0",
                    "  estimation:",
                    "    fit_intervention_model: true",
                    "    beta_mask_pre_intervention: false",
                    "    beta_mask_rescale: false",
                    "    fixed_scalar_params: {}",
                    "variants:",
                    "  - name: rank_0",
                ]
            ),
            encoding="utf-8",
        )
        predictive_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    "  num_samples: 4",
                    "  gibbs_sweeps: 1",
                    "  seed: 0",
                    "runs:",
                    "  - name: default",
                    "  - name: longer",
                    "    num_samples: 5",
                    "    gibbs_sweeps: 2",
                ]
            ),
            encoding="utf-8",
        )
        target_pairs_path.write_text(
            "\n".join(
                [
                    "experiment_name,source_type,variant_name",
                    "smoke_rank_0,truth,",
                    "smoke_rank_0,fit,rank_0",
                ]
            ),
            encoding="utf-8",
        )

        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        fit_manifest = run_fits(generation_manifest, fits_spec_path, overwrite=True)

        experiment_root = self.root / "generated" / "smoke_rank_0"
        truth_bundle = load_truth_parameter_bundle(experiment_root)
        fit_bundle = load_fit_parameter_bundle(experiment_root / "fits" / "rank_0", experiment_root)
        self.assertEqual(truth_bundle.field_matrix.shape, (4, 6))
        self.assertEqual(fit_bundle.field_matrix.shape, (4, 6))

        predictive_manifest = run_posterior_predictive(
            generation_manifest,
            fit_manifest,
            target_pairs_path,
            predictive_spec_path,
            overwrite=True,
        )

        truth_default_csv = (
            experiment_root
            / "posterior_predictive"
            / "truth"
            / "default"
            / "posterior_predictive_stats.csv"
        )
        truth_longer_csv = (
            experiment_root
            / "posterior_predictive"
            / "truth"
            / "longer"
            / "posterior_predictive_stats.csv"
        )
        fit_default_csv = (
            experiment_root
            / "posterior_predictive"
            / "fit_rank_0"
            / "default"
            / "posterior_predictive_stats.csv"
        )
        fit_longer_csv = (
            experiment_root
            / "posterior_predictive"
            / "fit_rank_0"
            / "longer"
            / "posterior_predictive_stats.csv"
        )
        summary_csv = experiment_root / "posterior_predictive_summary.csv"
        summary_md = experiment_root / "posterior_predictive_summary.md"
        winners_csv = self.root / "generated" / "best_posterior_predictive_by_experiment.csv"
        winners_md = self.root / "generated" / "best_posterior_predictive_by_experiment.md"

        self.assertTrue(truth_default_csv.exists())
        self.assertTrue(truth_longer_csv.exists())
        self.assertTrue(fit_default_csv.exists())
        self.assertTrue(fit_longer_csv.exists())
        self.assertTrue(summary_csv.exists())
        self.assertTrue(summary_md.exists())
        self.assertTrue(winners_csv.exists())
        self.assertTrue(winners_md.exists())

        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(sum(row["is_best"] == "True" for row in rows), 1)
        self.assertEqual(len(rows), 4)
        self.assertIn("mean_abs_zscore", rows[0])

        with winners_csv.open("r", encoding="utf-8", newline="") as handle:
            winner_rows = list(csv.DictReader(handle))
        self.assertEqual(len(winner_rows), 1)
        self.assertEqual(winner_rows[0]["experiment_name"], "smoke_rank_0")
        self.assertIn("coverage_rate", winner_rows[0])
        self.assertEqual(
            Path(predictive_manifest),
            self.root / "generated" / "posterior_predictive_manifest.csv",
        )


if __name__ == "__main__":
    unittest.main()
