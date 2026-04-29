"""Small regression tests for the minimal latent-only MPLE pipeline."""

from __future__ import annotations

import argparse
import shutil
import os
import subprocess
import sys
import unittest
from unittest import mock
import uuid
import csv
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.synthetic_data_generation import (
    generate_data,
    load_fixed_intervention_artifacts,
    simulate_outcomes_given_fixed_interventions,
)
from data.USCountyVaccination import (
    create_us_county_vaccination_experiments as uscounty_materializer,
)
from data.USCountyVaccination.experiment_artifacts import (
    RealizedBinaryArtifact,
    RealizedNetworkArtifact,
    assembled_panel_from_arrays,
    create_config as create_us_county_config,
    realized_intervention_name,
    realized_network_name,
    realized_outcome_name,
    save_experiment as save_us_county_experiment,
    shared_panel_name,
    write_realized_binary_artifact,
    write_realized_network_artifact,
    write_shared_panel_artifacts,
)
from intervention_utils import load_saved_intervention_context
from io_utils import io_path
from loading_utils import (
    OutcomeParameterBundle,
    load_experiment_panel_context,
    load_fit_parameter_bundle,
    load_truth_parameter_bundle,
    save_estimated_parameter_bundle,
)
from mple import (
    _build_fit_eval_context,
    _compute_h_x,
    _evaluate_factorized_loss,
    _evaluate_full_field_loss,
    _evaluate_scalar_only_loss,
    _project_node_factor_columns_to_l2_ball,
    fit_mple,
    pseudo_nll,
)
from model_utils import (
    ModelArtifacts,
    build_fit_model_artifacts,
    build_synthetic_field,
    compose_interaction_matrix,
    compose_latent_field_matrix,
    get_xi,
    _smooth_time_trend,
    interaction_effect,
    interaction_matrix_infinity_norm,
    latent_field_bound_norm,
    load_model_artifacts,
    SYNTHETIC_FIELD_MODE_LOW_RANK_PLUS_EARLY_TREATMENT_CONFOUNDING,
    load_true_parameters,
    parameter_names,
    project_latent_field,
    save_model_artifacts,
    unpack_theta,
)
from pipeline_specs import read_csv_manifest, validate_fits_spec
from posterior_predictive_utils import (
    compute_panel_statistics,
    simulate_outcomes_for_bundle,
    summarize_predictive_statistics,
)
from report_posterior_predictive import (
    collect_predictive_rows,
    group_and_rank_predictive_rows,
    refresh_and_write_posterior_predictive_reports,
    write_intervention_summaries,
)
from report_parameter_recovery_detailed import (
    collect_fit_rows,
    group_and_rank_fit_rows,
    write_fit_reports,
)
from run_fit_pipeline import (
    build_fit_config,
    infer_panel_dimensions,
    refresh_fit_manifest,
    run_fit_request,
    run_fits,
    write_fit_requests,
)
from run_generation_pipeline import (
    refresh_generation_manifest,
    run_generation,
    run_generation_request,
    write_generation_requests,
)
from run_intervention_library import run_intervention_library
from posterior_predictive_job_utils import (
    index_generation_rows,
    resolve_fit_lookup,
    resolve_target_pairs,
)
from run_posterior_predictive import run_posterior_predictive


def base_config() -> object:
    return OmegaConf.create(
        {
            "global_params": {
                "N": 4,
                "T": 3,
                "s": 1,
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
                "gibbs_sweeps": 1,
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

    def test_generate_data_fixed_z_returns_supplied_panel_and_z0(self) -> None:
        config = base_config()
        config.generation_params.intervention_mode = "fixed_z"
        config.generation_params.gibbs_sweeps = 2
        x_0 = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
        z_0 = np.array([-1.0, -1.0, -1.0, -1.0], dtype=float)
        fixed_z = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
            ],
            dtype=float,
        )
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        artifacts = build_synthetic_field(config, gamma)
        seed = 123

        generated_x, generated_z, returned_z_0 = generate_data(
            config,
            artifacts,
            x_0,
            np.random.default_rng(seed),
            fixed_z=fixed_z,
            z_0=z_0,
        )
        expected_x = simulate_outcomes_given_fixed_interventions(
            x_0=x_0,
            z=fixed_z,
            field_matrix=artifacts.field_matrix,
            interaction_matrix=compose_interaction_matrix(
                get_xi(config), artifacts.gamma_matrix
            ),
            beta=float(config.estimation_params.beta),
            eta=float(config.estimation_params.eta),
            rng=np.random.default_rng(seed),
            gibbs_sweeps=int(config.generation_params.gibbs_sweeps),
        )

        self.assertTrue(np.array_equal(generated_z, fixed_z))
        self.assertTrue(np.array_equal(returned_z_0, z_0))
        self.assertTrue(np.array_equal(generated_x, expected_x))

    def test_generate_data_generated_z_respects_pre_intervention_steps(self) -> None:
        config = base_config()
        config.global_params.T = 5
        config.global_params.s = 2
        x_0 = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        artifacts = build_synthetic_field(config, gamma)

        _, z, z_0 = generate_data(
            config,
            artifacts,
            x_0,
            np.random.default_rng(321),
        )

        self.assertTrue(np.array_equal(z[:2, :], -np.ones((2, 4), dtype=float)))
        self.assertTrue(np.array_equal(z_0, np.zeros(4, dtype=float)))

    def test_posterior_predictive_matches_fixed_z_generation(self) -> None:
        config = base_config()
        config.generation_params.intervention_mode = "fixed_z"
        config.generation_params.gibbs_sweeps = 2
        x_0 = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
        fixed_z = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
            ],
            dtype=float,
        )
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        artifacts = build_synthetic_field(config, gamma)
        bundle = OutcomeParameterBundle(
            source_type="truth",
            source_name="truth",
            beta=float(config.estimation_params.beta),
            xi=float(config.estimation_params.xi),
            eta=float(config.estimation_params.eta),
            beta_mask_pre_s=False,
            latent_rank=int(artifacts.latent_rank),
            t_steps=int(config.global_params.T),
            field_matrix=np.asarray(artifacts.field_matrix, dtype=float),
            gamma_matrix=artifacts.gamma_matrix,
        )
        seed = 777

        generated_x, _, _ = generate_data(
            config,
            artifacts,
            x_0,
            np.random.default_rng(seed),
            fixed_z=fixed_z,
        )
        predictive_x = simulate_outcomes_for_bundle(
            bundle,
            x_0=x_0,
            z=fixed_z,
            gibbs_sweeps=int(config.generation_params.gibbs_sweeps),
            seed=seed,
        )

        self.assertTrue(np.array_equal(generated_x, predictive_x))

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
        self.assertEqual(artifacts.latent_rank, 2)
        field_matrix = np.asarray(artifacts.field_matrix, dtype=float)
        self.assertEqual(field_matrix.shape, (3, 4))
        self.assertLessEqual(np.linalg.matrix_rank(field_matrix), 2)
        self.assertLessEqual(latent_field_bound_norm(field_matrix), 1.0 + 1e-8)

    def test_generated_latent_field_uses_target_rms_scaling(self) -> None:
        config = base_config()
        config.global_params.latent_rank = 2
        config.global_params.B = 0.5
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        artifacts = build_synthetic_field(config, gamma)
        field_matrix = np.asarray(artifacts.field_matrix, dtype=float)
        target_rms = 0.4 * float(config.global_params.B)

        self.assertLessEqual(np.linalg.matrix_rank(field_matrix), 2)
        self.assertLessEqual(
            float(np.sqrt(np.mean(field_matrix**2))),
            target_rms + 1e-12,
        )
        self.assertLessEqual(
            latent_field_bound_norm(field_matrix),
            float(config.global_params.B) + 1e-12,
        )

    def test_node_bias_plus_smooth_time_drift_field_is_rank_two(self) -> None:
        config = base_config()
        config.global_params.B = 1.0
        config.global_params.latent_rank = 2
        config.global_params.field_mode = "node_bias_plus_smooth_time_drift"
        config.global_params.field_params = {
            "node_bias_scale": 1.0,
            "drift_scale": 0.4,
            "time_trend_sharpness": 2.0,
        }
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        artifacts = build_synthetic_field(config, gamma)
        field_matrix = np.asarray(artifacts.field_matrix, dtype=float)
        trend = _smooth_time_trend(int(config.global_params.T), sharpness=2.0)

        self.assertEqual(artifacts.latent_rank, 2)
        self.assertEqual(field_matrix.shape, (3, 4))
        self.assertLessEqual(np.linalg.matrix_rank(field_matrix), 2)
        self.assertLessEqual(
            latent_field_bound_norm(field_matrix),
            float(config.global_params.B) + 1e-12,
        )
        self.assertAlmostEqual(float(np.mean(trend)), 0.0, places=12)
        self.assertAlmostEqual(float(np.sqrt(np.mean(trend**2))), 1.0, places=12)
        self.assertTrue(np.all(np.diff(trend) >= -1e-12))

    def test_low_rank_plus_early_treatment_confounding_orders_county_bias(self) -> None:
        config = base_config()
        config.global_params.B = 1.0
        config.global_params.N = 4
        config.global_params.T = 4
        config.global_params.latent_rank = 0
        config.global_params.field_mode = (
            SYNTHETIC_FIELD_MODE_LOW_RANK_PLUS_EARLY_TREATMENT_CONFOUNDING
        )
        config.global_params.field_params = {
            "confounding_bias_scale": 1.0,
            "untreated_score_value": 0.0,
        }
        z = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, -1.0],
            ],
            dtype=float,
        )
        temp_root = REPO_ROOT / "experiments" / f".tmp_confounding_{uuid.uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            panel_path = temp_root / "panel_data.npz"
            z0_path = temp_root / "z_0.npy"
            np.savez(panel_path, z=z)
            np.save(z0_path, np.zeros(4, dtype=float))
            config.generation_params.fixed_z_source = OmegaConf.create(
                {
                    "panel_path": str(panel_path),
                    "z0_path": str(z0_path),
                }
            )
            gamma = np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
            artifacts = build_synthetic_field(config, gamma)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        field_matrix = np.asarray(artifacts.field_matrix, dtype=float)
        self.assertEqual(field_matrix.shape, (4, 4))
        self.assertEqual(artifacts.latent_rank, 1)
        self.assertLessEqual(
            latent_field_bound_norm(field_matrix),
            float(config.global_params.B) + 1e-12,
        )
        column_means = np.mean(field_matrix, axis=0)
        self.assertLessEqual(column_means[0], column_means[1] + 1e-12)
        self.assertLessEqual(column_means[1], column_means[2] + 1e-12)
        self.assertLessEqual(column_means[2], column_means[3] + 1e-12)

    def test_generation_spec_includes_us_county_confounding_experiment(self) -> None:
        from pipeline_specs import expand_named_entries

        experiments = expand_named_entries(REPO_ROOT / "data" / "configs" / "generation_spec.yaml", "experiments")
        confounding_spec = next(
            experiment
            for experiment in experiments
            if experiment["name"] == "hybrid_us_county_intervention_uscounty_graph_confounding"
        )
        self.assertEqual(
            confounding_spec["truth"]["field_mode"],
            SYNTHETIC_FIELD_MODE_LOW_RANK_PLUS_EARLY_TREATMENT_CONFOUNDING,
        )
        self.assertEqual(int(confounding_spec["truth"]["latent_rank"]), 20)
        self.assertIn("confounding_bias_scale", confounding_spec["truth"]["field_params"])

    def test_generation_pipeline_smoke_with_confounding_field_mode(self) -> None:
        root = REPO_ROOT / "experiments" / f".tmp_gen_confounding_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            fixed_panel_path = root / "fixed_panel.npz"
            fixed_z0_path = root / "fixed_z0.npy"
            gamma_path = root / "gamma.npy"
            spec_path = root / "generation_spec.yaml"
            fits_spec_path = root / "fits_spec.yaml"

            z = np.array(
                [
                    [-1.0, -1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0, -1.0],
                    [1.0, 1.0, -1.0, -1.0],
                    [1.0, 1.0, 1.0, -1.0],
                ],
                dtype=float,
            )
            gamma = np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
            np.savez(fixed_panel_path, z=z)
            np.save(fixed_z0_path, np.zeros(4, dtype=float))
            np.save(gamma_path, gamma)

            spec_path.write_text(
                "\n".join(
                    [
                        "base:",
                        f"  experiment_root: {root.as_posix()}/generated",
                        f"  manifest_path: {root.as_posix()}/generated/generation_manifest.csv",
                        "  dimensions:",
                        "    N: 4",
                        "    T: 4",
                        "    s: 0",
                        "  generation:",
                        "    gibbs_sweeps: 1",
                        "    seed: 7",
                        "  x0:",
                        "    generator: bernoulli",
                        "    params:",
                        "      p: 0.5",
                        "      fixed_val: null",
                        "  graph:",
                        "    source: fixed_artifact",
                        "    artifact:",
                        f"      gamma_path: {gamma_path.as_posix()}",
                        "      node_index_path: null",
                        f"      artifact_dir: {root.as_posix()}",
                        "      network_name: test_graph",
                        "      trim_scope: test",
                        "  intervention:",
                        "    source: fixed_artifact",
                        "    artifact:",
                        f"      panel_path: {fixed_panel_path.as_posix()}",
                        f"      z0_path: {fixed_z0_path.as_posix()}",
                        f"      artifact_dir: {root.as_posix()}",
                        "      shared_panel_dir: null",
                        "      outcome_code: null",
                        "      intervention_code: test_intervention",
                        "      lag_code: test",
                        "      trim_scope: test",
                        "  truth:",
                        "    B: 1.0",
                        "    latent_rank: 2",
                        f"    field_mode: {SYNTHETIC_FIELD_MODE_LOW_RANK_PLUS_EARLY_TREATMENT_CONFOUNDING}",
                        "    field_params:",
                        "      confounding_bias_scale: 0.75",
                        "      untreated_score_value: 0.0",
                        "    scalars:",
                        "      beta: 0.2",
                        "      xi: 0.1",
                        "      eta: 0.05",
                        "      zeta: -0.1",
                        "      psi: 0.2",
                        "experiments:",
                        "  - name: confounding_smoke",
                    ]
                ),
                encoding="utf-8",
            )
            fits_spec_path.write_text(
                "\n".join(
                    [
                        "base:",
                        "  fit_root_name: fits",
                        f"  fit_manifest_path: {root.as_posix()}/generated/fit_manifest.csv",
                        "  optimizer:",
                        "    steps: 5",
                        "    tol: 1.0e-6",
                        "    seed: 0",
                        "  B: 1.0",
                        "  latent_rank: 0",
                        "  estimation:",
                        "    fixed_scalar_params: {}",
                        "variants:",
                        "  - name: rank_0",
                    ]
                ),
                encoding="utf-8",
            )

            generation_manifest = run_generation(spec_path, overwrite=True)
            fit_manifest = run_fits(generation_manifest, fits_spec_path, overwrite=True)
            experiment_root = root / "generated" / "confounding_smoke"

            self.assertTrue((experiment_root / "panel_data.npz").exists())
            self.assertTrue((experiment_root / "field_artifacts.npz").exists())
            self.assertTrue((experiment_root / "fits" / "rank_0" / "mple_summary.csv").exists())
            self.assertEqual(
                Path(fit_manifest),
                root / "generated" / "fit_manifest.csv",
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_latent_field_projection_bounds_max_entry_not_row_sum(self) -> None:
        node_factors = np.array([[2.0], [2.0]])
        time_factors = np.array([[2.0], [2.0]])

        projected_nodes, projected_times = project_latent_field(
            node_factors,
            time_factors,
            bound=1.0,
        )
        field_matrix = compose_latent_field_matrix(projected_nodes, projected_times)

        self.assertLessEqual(latent_field_bound_norm(field_matrix), 1.0 + 1e-12)
        self.assertGreater(float(np.linalg.norm(field_matrix, ord=np.inf)), 1.0)

    def test_fit_report_latent_diagnostics_uses_max_abs_entry_names(self) -> None:
        from report_parameter_recovery_detailed import latent_diagnostics

        fit_root = REPO_ROOT / "experiments" / f".tmp_latent_diag_{uuid.uuid4().hex}"
        fit_root.mkdir(parents=True, exist_ok=True)
        try:
            estimated_field = np.array([[2.0, -1.0], [2.0, -1.0]], dtype=float)
            true_field = np.array([[0.25, -0.75], [0.5, -0.5]], dtype=float)
            np.savez(fit_root / "estimated_field_artifacts.npz", field_matrix=estimated_field)
            np.savez(fit_root / "true_field_artifacts.npz", field_matrix=true_field)

            row = latent_diagnostics(fit_root)

            self.assertEqual(row["estimated_field_max_abs_entry"], 2.0)
            self.assertEqual(row["true_field_max_abs_entry"], 0.75)
            self.assertNotIn("estimated_field_inf_norm", row)
            self.assertNotIn("true_field_inf_norm", row)
            self.assertGreater(float(np.linalg.norm(estimated_field, ord=np.inf)), 2.0)
        finally:
            shutil.rmtree(fit_root, ignore_errors=True)

    def test_fit_report_latent_diagnostics_reads_legacy_inf_norm_names(self) -> None:
        from report_parameter_recovery_detailed import latent_diagnostics

        fit_root = REPO_ROOT / "experiments" / f".tmp_legacy_latent_diag_{uuid.uuid4().hex}"
        fit_root.mkdir(parents=True, exist_ok=True)
        try:
            (fit_root / "mple_summary.csv").write_text(
                "\n".join(
                    [
                        "category,name,estimate,true,squared_error",
                        "latent_diagnostic,estimated_field_inf_norm,1.25,,",
                        "latent_diagnostic,true_field_inf_norm,0.75,,",
                    ]
                ),
                encoding="utf-8",
            )

            row = latent_diagnostics(fit_root)

            self.assertEqual(row["estimated_field_max_abs_entry"], 1.25)
            self.assertEqual(row["true_field_max_abs_entry"], 0.75)
        finally:
            shutil.rmtree(fit_root, ignore_errors=True)

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
            self.assertEqual(theta.shape[0], 3 * 4 + 3)
            with np.load(root / "field_artifacts.npz", allow_pickle=False) as data:
                self.assertIn("field_matrix", data)
                self.assertIn("latent_rank", data)
                self.assertIn("t_steps", data)
                self.assertNotIn("field_mode", data)
                self.assertNotIn("node_factors", data)
                self.assertNotIn("time_factors", data)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_scalar_only_theta_preserves_unconstrained_scalars_and_interaction(
        self,
    ) -> None:
        gamma = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=1,
            latent_rank=0,
            optimizer_mode="no_external_field",
        )
        parts = unpack_theta(np.array([5.0, 4.0, -3.5], dtype=float), artifacts)
        self.assertAlmostEqual(float(parts["beta"]), 5.0)
        self.assertAlmostEqual(float(parts["xi"]), 4.0)
        self.assertAlmostEqual(float(parts["eta"]), -3.5)
        interaction = compose_interaction_matrix(
            float(parts["xi"]), artifacts.gamma_matrix
        )
        self.assertGreater(
            interaction_matrix_infinity_norm(interaction),
            1.0,
        )

    def test_pseudo_nll_gradient_matches_low_rank_factor_loss_scaling(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
        )
        theta = np.array(
            [0.1, -0.2, 0.05, 0.15, 0.3, -0.25, 0.2],
            dtype=float,
        )
        direction = np.array(
            [0.2, -0.1, 0.3, -0.4, 0.5, 0.1, -0.2],
            dtype=float,
        )
        direction /= np.linalg.norm(direction)
        kwargs = dict(
            x=x,
            z=z,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            fixed_scalar_params={},
        )

        _, grad = pseudo_nll(theta=theta, **kwargs)
        eps = 1.0e-6
        loss_plus, _ = pseudo_nll(theta=theta + eps * direction, **kwargs)
        loss_minus, _ = pseudo_nll(theta=theta - eps * direction, **kwargs)
        finite_difference = (loss_plus - loss_minus) / (2.0 * eps)

        self.assertAlmostEqual(
            float(grad @ direction),
            float(finite_difference),
            places=8,
        )

    def test_specialized_scalar_only_kernel_matches_pseudo_nll(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=0,
            optimizer_mode="no_external_field",
        )
        theta = np.array([0.3, -0.25, 0.2], dtype=float)
        context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
        )

        ref_loss, ref_grad = pseudo_nll(
            x=x,
            z=z,
            theta=theta,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            fixed_scalar_params={},
        )
        kernel_loss, kernel_grad = _evaluate_scalar_only_loss(theta, context)

        self.assertAlmostEqual(kernel_loss, ref_loss, places=12)
        self.assertTrue(np.allclose(kernel_grad, ref_grad))

    def test_specialized_full_field_kernel_matches_pseudo_nll(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=0,
            optimizer_mode="nuclear_norm",
        )
        theta = np.array([0.1, -0.2, 0.05, 0.15, 0.3, -0.25, 0.2], dtype=float)
        context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
        )

        ref_loss, ref_grad = pseudo_nll(
            x=x,
            z=z,
            theta=theta,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            fixed_scalar_params={},
        )
        field_matrix = theta[:4].reshape(2, 2)
        kernel_loss, residual, scalar_grad = _evaluate_full_field_loss(
            field_matrix,
            context,
            free_scalar_values=theta[4:],
        )
        kernel_grad = np.concatenate(
            [(residual / x.size).reshape(-1), scalar_grad]
        )

        self.assertAlmostEqual(kernel_loss, ref_loss, places=12)
        self.assertTrue(np.allclose(kernel_grad, ref_grad))

    def test_specialized_factorized_kernel_matches_pseudo_nll(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
        )
        theta = np.array(
            [0.1, -0.2, 0.05, 0.15, 0.3, -0.25, 0.2],
            dtype=float,
        )
        context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
        )

        ref_loss, ref_grad = pseudo_nll(
            x=x,
            z=z,
            theta=theta,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            fixed_scalar_params={},
        )
        node_factors = theta[:2].reshape(2, 1)
        time_factors = theta[2:4].reshape(2, 1)
        kernel_loss, _, time_grad, node_grad, scalar_grad = _evaluate_factorized_loss(
            time_factors,
            node_factors,
            context,
            free_scalar_values=theta[4:],
        )
        kernel_grad = np.concatenate(
            [node_grad.reshape(-1), time_grad.reshape(-1), scalar_grad]
        )

        self.assertAlmostEqual(kernel_loss, ref_loss, places=12)
        self.assertTrue(np.allclose(kernel_grad, ref_grad))

    def test_specialized_scalar_only_kernel_matches_masked_pseudo_nll(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=3,
            latent_rank=0,
            optimizer_mode="no_external_field",
        )
        theta = np.array([0.3, -0.25, 0.2], dtype=float)
        context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
            s=1,
            beta_mask_pre_s=True,
        )

        ref_loss, ref_grad = pseudo_nll(
            x=x,
            z=z,
            theta=theta,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            fixed_scalar_params={},
            s=1,
            beta_mask_pre_s=True,
        )
        kernel_loss, kernel_grad = _evaluate_scalar_only_loss(theta, context)

        self.assertAlmostEqual(kernel_loss, ref_loss, places=12)
        self.assertTrue(np.allclose(kernel_grad, ref_grad))

    def test_specialized_full_field_kernel_matches_masked_pseudo_nll(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=3,
            latent_rank=0,
            optimizer_mode="nuclear_norm",
        )
        theta = np.array(
            [0.1, -0.2, 0.05, 0.15, -0.1, 0.2, 0.3, -0.25, 0.2],
            dtype=float,
        )
        context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
            s=1,
            beta_mask_pre_s=True,
        )

        ref_loss, ref_grad = pseudo_nll(
            x=x,
            z=z,
            theta=theta,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            fixed_scalar_params={},
            s=1,
            beta_mask_pre_s=True,
        )
        field_matrix = theta[:6].reshape(3, 2)
        kernel_loss, residual, scalar_grad = _evaluate_full_field_loss(
            field_matrix,
            context,
            free_scalar_values=theta[6:],
        )
        kernel_grad = np.concatenate(
            [(residual / x.size).reshape(-1), scalar_grad]
        )

        self.assertAlmostEqual(kernel_loss, ref_loss, places=12)
        self.assertTrue(np.allclose(kernel_grad, ref_grad))

    def test_specialized_factorized_kernel_matches_masked_pseudo_nll(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=3,
            latent_rank=1,
        )
        theta = np.array(
            [0.1, -0.2, 0.05, 0.15, -0.1, 0.3, -0.25, 0.2],
            dtype=float,
        )
        context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
            s=1,
            beta_mask_pre_s=True,
        )

        ref_loss, ref_grad = pseudo_nll(
            x=x,
            z=z,
            theta=theta,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            fixed_scalar_params={},
            s=1,
            beta_mask_pre_s=True,
        )
        node_factors = theta[:2].reshape(2, 1)
        time_factors = theta[2:5].reshape(3, 1)
        kernel_loss, _, time_grad, node_grad, scalar_grad = _evaluate_factorized_loss(
            time_factors,
            node_factors,
            context,
            free_scalar_values=theta[5:],
        )
        kernel_grad = np.concatenate(
            [node_grad.reshape(-1), time_grad.reshape(-1), scalar_grad]
        )

        self.assertAlmostEqual(kernel_loss, ref_loss, places=12)
        self.assertTrue(np.allclose(kernel_grad, ref_grad))

    def test_beta_mask_pre_s_changes_beta_effect_only_after_s(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
            s=1,
            beta_mask_pre_s=True,
        )

        h_without_beta = _compute_h_x(
            np.zeros_like(x),
            {"beta": 0.0, "xi": 0.0, "eta": 0.0},
            context,
        )
        h_with_beta = _compute_h_x(
            np.zeros_like(x),
            {"beta": 2.0, "xi": 0.0, "eta": 0.0},
            context,
        )

        self.assertTrue(np.allclose(h_without_beta[:1], h_with_beta[:1]))
        self.assertFalse(np.allclose(h_without_beta[1:], h_with_beta[1:]))

    def test_beta_mask_pre_s_leaves_xi_and_eta_active_pre_s(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
            s=1,
            beta_mask_pre_s=True,
        )

        h_without_xi_eta = _compute_h_x(
            np.zeros_like(x),
            {"beta": 0.0, "xi": 0.0, "eta": 0.0},
            context,
        )
        h_with_xi_eta = _compute_h_x(
            np.zeros_like(x),
            {"beta": 0.0, "xi": 0.5, "eta": -0.75},
            context,
        )

        self.assertFalse(np.allclose(h_without_xi_eta[:1], h_with_xi_eta[:1]))
        self.assertFalse(np.allclose(h_without_xi_eta[1:], h_with_xi_eta[1:]))

    def test_pseudo_nll_gradient_matches_outcome_only_loss_scaling(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=0,
            optimizer_mode="nuclear_norm",
        )
        theta = np.array([0.1, -0.2, 0.05, 0.15, 0.3, -0.25, 0.2], dtype=float)
        direction = np.array([0.2, -0.1, 0.3, -0.4, 0.5, 0.1, -0.2], dtype=float)
        direction /= np.linalg.norm(direction)
        kwargs = dict(
            x=x,
            z=z,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            fixed_scalar_params={},
        )

        _, grad = pseudo_nll(theta=theta, **kwargs)
        eps = 1.0e-6
        loss_plus, _ = pseudo_nll(theta=theta + eps * direction, **kwargs)
        loss_minus, _ = pseudo_nll(theta=theta - eps * direction, **kwargs)
        finite_difference = (loss_plus - loss_minus) / (2.0 * eps)

        self.assertAlmostEqual(
            float(grad @ direction),
            float(finite_difference),
            places=8,
        )

    def test_fit_mple_uses_s_when_beta_mask_pre_s_is_enabled(self) -> None:
        x = np.ones((2, 2), dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.zeros(2, dtype=float)
        gamma = np.zeros((2, 2), dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=0,
            optimizer_mode="no_external_field",
        )
        fixed_scalars = {"xi": 0.0, "eta": 0.0}
        param_keys = parameter_names(
            artifacts,
            fixed_scalar_params=fixed_scalars,
        )

        theta_no_mask, _, _ = fit_mple(
            x,
            z,
            x_0=x_0,
            s=0,
            param_names=param_keys,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=25,
            tol=1.0e-8,
            seed=0,
            verbose_every=0,
            fixed_scalar_params=fixed_scalars,
            beta_mask_pre_s=True,
        )
        theta_masked, _, _ = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=param_keys,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=25,
            tol=1.0e-8,
            seed=0,
            verbose_every=0,
            fixed_scalar_params=fixed_scalars,
            beta_mask_pre_s=True,
        )

        self.assertEqual(theta_no_mask.shape, (1,))
        self.assertEqual(theta_masked.shape, (1,))
        self.assertGreater(float(theta_masked[0]), float(theta_no_mask[0]))

    def test_fit_mple_uses_pymanopt_multistart_for_low_rank(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
        )
        param_keys = parameter_names(artifacts)
        theta_hat, loss_history, result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=param_keys,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=2,
            tol=1.0e-6,
            seed=11,
            verbose_every=0,
            n_starts=2,
        )

        self.assertEqual(theta_hat.shape, (len(param_keys),))
        self.assertTrue(np.isfinite(loss_history[-1]))
        self.assertEqual(result["n_starts"], 2)
        self.assertEqual(result["optimizer"], "pymanopt_conjugate_gradient")
        self.assertEqual(len(result["start_summaries"]), 2)
        self.assertIn(int(result["best_start"]), (0, 1))
        self.assertIn("mple_history", result)
        self.assertIn("penalized_history", result)
        self.assertEqual(
            sorted(result["start_summaries"][0].keys()),
            sorted(
                [
                    "start_index",
                    "seed",
                    "initialization_kind",
                    "initial_mple_loss",
                    "initial_penalized_objective",
                    "final_mple_loss",
                    "final_penalized_objective",
                    "iterations",
                    "cost_evaluations",
                    "success",
                    "message",
                ]
            ),
        )

    def test_manifold_frobenius_penalty_shrinks_field_norm(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
        )
        fixed_scalars = {"beta": 0.0, "xi": 0.0, "eta": 0.0}
        param_keys = parameter_names(
            artifacts,
            fixed_scalar_params=fixed_scalars,
        )
        low_penalty_theta, _, low_penalty_result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=param_keys,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=10,
            tol=1.0e-8,
            seed=11,
            verbose_every=0,
            n_starts=1,
            fixed_scalar_params=fixed_scalars,
            lambda_frobenius=0.0,
        )
        high_penalty_theta, _, high_penalty_result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=param_keys,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=10,
            tol=1.0e-8,
            seed=11,
            verbose_every=0,
            n_starts=1,
            fixed_scalar_params=fixed_scalars,
            lambda_frobenius=1.0,
        )

        self.assertEqual(low_penalty_theta.shape, high_penalty_theta.shape)
        self.assertEqual(
            high_penalty_result["optimizer"],
            "pymanopt_conjugate_gradient",
        )
        self.assertAlmostEqual(float(high_penalty_result["lambda_frobenius"]), 1.0)
        self.assertLess(
            float(high_penalty_result["frobenius_norm"]),
            float(low_penalty_result["frobenius_norm"]),
        )
        self.assertAlmostEqual(
            float(high_penalty_result["normalized_frobenius_norm"]),
            float(high_penalty_result["frobenius_norm"]) / 2.0,
        )
        self.assertAlmostEqual(
            float(high_penalty_result["squared_normalized_frobenius_norm"]),
            float(high_penalty_result["frobenius_norm"]) ** 2 / 4.0,
        )
        self.assertAlmostEqual(
            float(high_penalty_result["frobenius_penalty_normalizer"]),
            4.0,
        )
        self.assertGreaterEqual(
            float(high_penalty_result["final_penalized_objective"]),
            float(high_penalty_result["final_mple_loss"]),
        )

    def test_parameter_names_and_unpack_respect_fixed_scalars(self) -> None:
        artifacts = ModelArtifacts(
            gamma_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            t_steps=1,
            latent_rank=0,
        )
        theta = np.array([0.1, 0.25], dtype=float)
        names = parameter_names(
            artifacts,
            fixed_scalar_params={"beta": 0.0},
        )
        self.assertEqual(names, ["xi", "eta"])
        parts = unpack_theta(
            theta,
            artifacts,
            fixed_scalar_params={"beta": 0.0},
        )
        self.assertEqual(parts["beta"], 0.0)
        self.assertAlmostEqual(parts["xi"], 0.1)
        self.assertAlmostEqual(parts["eta"], 0.25)
        self.assertEqual(parts["node_factors"].shape, (2, 0))
        self.assertEqual(parts["time_factors"].shape, (1, 0))
        with self.assertRaises(ValueError):
            parameter_names(artifacts, fixed_scalar_params={"psi": 0.3})

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

    def test_nuclear_norm_fit_mode_uses_full_field_parameterization(self) -> None:
        config = base_config()
        config.global_params.optimizer_mode = "nuclear_norm"
        config.global_params.latent_rank = 99
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

        artifacts = build_fit_model_artifacts(config, gamma)
        names = parameter_names(artifacts)

        self.assertEqual(artifacts.optimizer_mode, "nuclear_norm")
        self.assertEqual(artifacts.latent_rank, 0)
        self.assertEqual(len([name for name in names if name.startswith("F::")]), 6)
        self.assertEqual(names[-3:], ["beta", "xi", "eta"])

    def test_nuclear_norm_fit_shrinks_singular_values_without_field_clipping(
        self,
    ) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=0,
            optimizer_mode="nuclear_norm",
        )
        param_keys = parameter_names(artifacts)

        low_penalty_theta, _, low_penalty_result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=param_keys,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=10,
            tol=0.0,
            seed=11,
            verbose_every=0,
            lambda_nuclear=0.0,
        )
        high_penalty_theta, _, high_penalty_result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=param_keys,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=10,
            tol=0.0,
            seed=11,
            verbose_every=0,
            lambda_nuclear=0.5,
        )

        low_field = unpack_theta(
            low_penalty_theta,
            artifacts,
        )["field_matrix"]
        high_field = unpack_theta(
            high_penalty_theta,
            artifacts,
        )["field_matrix"]
        self.assertGreater(latent_field_bound_norm(low_field), 0.25)
        self.assertLessEqual(
            float(np.linalg.svd(high_field, compute_uv=False).sum()),
            float(np.linalg.svd(low_field, compute_uv=False).sum()) + 1e-12,
        )
        self.assertEqual(low_penalty_result["optimizer_mode"], "nuclear_norm")
        self.assertEqual(high_penalty_result["optimizer_mode"], "nuclear_norm")
        self.assertAlmostEqual(
            float(high_penalty_result["nuclear_norm_normalizer"]), 2.0
        )
        self.assertAlmostEqual(
            float(high_penalty_result["normalized_nuclear_norm"]),
            float(high_penalty_result["nuclear_norm"]) / 2.0,
        )

    def test_alternative_low_rank_uses_generic_bookkeeping(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
            optimizer_mode="alternating_latent_rank",
        )
        param_keys = parameter_names(artifacts)

        theta_hat, loss_history, result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=param_keys,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=4,
            tol=1.0e-8,
            seed=7,
            verbose_every=0,
            n_starts=2,
            lambda_uv_ridge=0.1,
        )

        self.assertEqual(theta_hat.shape, (len(param_keys),))
        self.assertTrue(np.isfinite(loss_history[-1]))
        self.assertEqual(result["optimizer_mode"], "alternating_latent_rank")
        self.assertEqual(result["optimizer"], "alternating_low_rank")
        self.assertEqual(result["n_starts"], 2)
        self.assertEqual(len(result["start_summaries"]), 2)
        self.assertIn("mple_history", result)
        self.assertIn("penalized_history", result)
        self.assertAlmostEqual(float(result["lambda_uv_ridge"]), 0.1)

    def test_alternating_low_rank_penalized_history_decreases(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
            optimizer_mode="alternating_latent_rank",
        )
        _, _, result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=parameter_names(artifacts),
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=4,
            tol=1.0e-8,
            seed=7,
            verbose_every=0,
            n_starts=1,
            lambda_uv_ridge=0.1,
        )
        history = list(result["penalized_history"])

        self.assertGreaterEqual(len(history), 2)
        self.assertLessEqual(history[-1], history[0])

    def test_project_node_factor_columns_to_l2_ball(self) -> None:
        node_factors = np.array(
            [
                [0.3, 3.0, 0.0],
                [0.4, 4.0, 0.0],
            ],
            dtype=float,
        )

        projected = _project_node_factor_columns_to_l2_ball(node_factors, 1.0)
        column_norms = np.linalg.norm(projected, axis=0)

        np.testing.assert_allclose(projected[:, 0], node_factors[:, 0])
        np.testing.assert_allclose(column_norms[1], 1.0)
        np.testing.assert_allclose(projected[:, 2], 0.0)

    def test_alternating_low_rank_enforces_v_column_l2_max(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
            optimizer_mode="alternating_latent_rank",
        )
        theta_hat, loss_history, result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=parameter_names(artifacts),
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=4,
            tol=1.0e-8,
            seed=7,
            verbose_every=0,
            n_starts=1,
            lambda_uv_ridge=0.1,
            v_column_l2_max=1.0,
        )
        theta_parts = unpack_theta(theta_hat, artifacts)
        column_norms = np.linalg.norm(
            np.asarray(theta_parts["node_factors"], dtype=float),
            axis=0,
        )

        self.assertTrue(np.isfinite(loss_history[-1]))
        self.assertTrue(np.isfinite(float(result["final_penalized_objective"])))
        self.assertTrue(np.all(column_norms <= 1.0 + 1e-12))

    def test_alternating_low_rank_omitted_v_column_l2_max_matches_none(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
            optimizer_mode="alternating_latent_rank",
        )
        common_kwargs = {
            "x": x,
            "z": z,
            "x_0": x_0,
            "s": 1,
            "param_names": parameter_names(artifacts),
            "artifacts": artifacts,
            "interaction_effect_x": interaction_effect(x, gamma),
            "steps": 4,
            "tol": 1.0e-8,
            "seed": 7,
            "verbose_every": 0,
            "n_starts": 1,
            "lambda_uv_ridge": 0.1,
        }

        theta_default, history_default, result_default = fit_mple(**common_kwargs)
        theta_none, history_none, result_none = fit_mple(
            **common_kwargs,
            v_column_l2_max=None,
        )

        np.testing.assert_allclose(theta_default, theta_none)
        np.testing.assert_allclose(history_default, history_none)
        self.assertAlmostEqual(
            float(result_default["final_penalized_objective"]),
            float(result_none["final_penalized_objective"]),
        )

    def test_build_fit_model_artifacts_rejects_nonpositive_alternative_rank(
        self,
    ) -> None:
        config = base_config()
        config.global_params.optimizer_mode = "alternating_latent_rank"
        config.global_params.latent_rank = 0
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

        with self.assertRaisesRegex(ValueError, "alternating_latent_rank"):
            build_fit_model_artifacts(config, gamma)

    def test_concurrent_low_rank_uses_generic_bookkeeping(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
            optimizer_mode="concurrent_latent_rank",
        )
        param_keys = parameter_names(artifacts)

        theta_hat, loss_history, result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=param_keys,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=8,
            tol=1.0e-8,
            seed=7,
            verbose_every=0,
            n_starts=2,
            lambda_uv_ridge=0.1,
        )

        self.assertEqual(theta_hat.shape, (len(param_keys),))
        self.assertTrue(np.isfinite(loss_history[-1]))
        self.assertEqual(result["optimizer_mode"], "concurrent_latent_rank")
        self.assertEqual(result["optimizer"], "scipy_lbfgsb_low_rank")
        self.assertEqual(result["n_starts"], 2)
        self.assertEqual(len(result["start_summaries"]), 2)
        self.assertIn("mple_history", result)
        self.assertIn("penalized_history", result)
        self.assertAlmostEqual(float(result["lambda_uv_ridge"]), 0.1)

    def test_concurrent_low_rank_penalized_history_decreases(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
            optimizer_mode="concurrent_latent_rank",
        )
        _, _, result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=parameter_names(artifacts),
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=8,
            tol=1.0e-8,
            seed=7,
            verbose_every=0,
            n_starts=1,
            lambda_uv_ridge=0.1,
        )
        history = list(result["penalized_history"])

        self.assertGreaterEqual(len(history), 2)
        self.assertLessEqual(history[-1], history[0])

    def test_concurrent_low_rank_penalty_changes_objective(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
            optimizer_mode="concurrent_latent_rank",
        )
        param_keys = parameter_names(artifacts)

        _, _, low_penalty_result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=param_keys,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=8,
            tol=1.0e-8,
            seed=7,
            verbose_every=0,
            n_starts=1,
            lambda_uv_ridge=0.0,
        )
        _, _, high_penalty_result = fit_mple(
            x,
            z,
            x_0=x_0,
            s=1,
            param_names=param_keys,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect(x, gamma),
            steps=8,
            tol=1.0e-8,
            seed=7,
            verbose_every=0,
            n_starts=1,
            lambda_uv_ridge=0.5,
        )

        self.assertLessEqual(
            float(high_penalty_result["final_mple_loss"]),
            float(high_penalty_result["final_penalized_objective"]) + 1e-12,
        )
        self.assertNotEqual(
            float(low_penalty_result["final_penalized_objective"]),
            float(high_penalty_result["final_penalized_objective"]),
        )

    def test_build_fit_model_artifacts_accepts_concurrent_rank(self) -> None:
        config = base_config()
        config.global_params.optimizer_mode = "concurrent_latent_rank"
        config.global_params.latent_rank = 2
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

        artifacts = build_fit_model_artifacts(config, gamma)
        self.assertEqual(artifacts.optimizer_mode, "concurrent_latent_rank")
        self.assertEqual(artifacts.latent_rank, 2)

    def test_build_fit_model_artifacts_rejects_nonpositive_concurrent_rank(
        self,
    ) -> None:
        config = base_config()
        config.global_params.optimizer_mode = "concurrent_latent_rank"
        config.global_params.latent_rank = 0
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

        with self.assertRaisesRegex(ValueError, "concurrent_latent_rank"):
            build_fit_model_artifacts(config, gamma)

    def test_build_fit_config_allows_uv_ridge_for_concurrent_mode(self) -> None:
        variant = {
            "name": "concurrent_rank_2",
            "optimizer": {"steps": 5, "tol": 1.0e-6, "seed": 3},
            "optimizer_mode": "concurrent_latent_rank",
            "latent_rank": 2,
            "lambda_uv_ridge": 0.25,
            "B": 1.0,
            "estimation": {"fixed_scalar_params": {}},
        }

        fit_config = build_fit_config(variant, {"N": 4, "T": 3, "s": 1})
        self.assertEqual(
            str(fit_config.global_params.optimizer_mode), "concurrent_latent_rank"
        )
        self.assertEqual(int(fit_config.global_params.latent_rank), 2)
        self.assertAlmostEqual(float(fit_config.global_params.lambda_uv_ridge), 0.25)

    def test_build_fit_config_rejects_nonpositive_v_column_l2_max(self) -> None:
        variant = {
            "name": "alternating_rank_1_bad_v_constraint",
            "optimizer": {"steps": 5, "tol": 1.0e-6, "seed": 3},
            "optimizer_mode": "alternating_latent_rank",
            "latent_rank": 1,
            "v_column_l2_max": 0.0,
            "B": 1.0,
            "estimation": {"fixed_scalar_params": {}},
        }

        with self.assertRaisesRegex(ValueError, "v_column_l2_max"):
            build_fit_config(variant, {"N": 4, "T": 3, "s": 1, "e": 2})

    def test_build_fit_config_defaults_beta_mask_pre_s_to_false(self) -> None:
        variant = {
            "name": "rank_0",
            "optimizer": {"steps": 5, "tol": 1.0e-6, "seed": 3},
            "optimizer_mode": "no_external_field",
            "latent_rank": 0,
            "estimation": {"fixed_scalar_params": {}},
        }

        fit_config = build_fit_config(variant, {"N": 4, "T": 3, "s": 1})
        self.assertFalse(bool(fit_config.estimation_params.beta_mask_pre_s))

    def test_build_fit_config_copies_beta_mask_pre_s(self) -> None:
        variant = {
            "name": "rank_0",
            "optimizer": {"steps": 5, "tol": 1.0e-6, "seed": 3},
            "optimizer_mode": "no_external_field",
            "latent_rank": 0,
            "estimation": {
                "fixed_scalar_params": {},
                "beta_mask_pre_s": True,
            },
        }

        fit_config = build_fit_config(variant, {"N": 4, "T": 3, "s": 1})
        self.assertTrue(bool(fit_config.estimation_params.beta_mask_pre_s))

    def test_build_fit_config_defaults_warm_start_settings(self) -> None:
        variant = {
            "name": "rank_0",
            "optimizer": {"steps": 5, "tol": 1.0e-6, "seed": 3},
            "optimizer_mode": "no_external_field",
            "latent_rank": 0,
            "estimation": {"fixed_scalar_params": {}},
        }

        fit_config = build_fit_config(variant, {"N": 4, "T": 3, "s": 1, "e": 2})
        self.assertEqual(
            OmegaConf.to_container(
                fit_config.estimation_params.warm_start_fixed_scalars, resolve=True
            ),
            {},
        )
        self.assertEqual(int(fit_config.estimation_params.warm_start_steps), 0)

    def test_build_fit_config_copies_warm_start_settings(self) -> None:
        variant = {
            "name": "rank_0",
            "optimizer": {"steps": 5, "tol": 1.0e-6, "seed": 3},
            "optimizer_mode": "no_external_field",
            "latent_rank": 0,
            "estimation": {
                "fixed_scalar_params": {},
                "warm_start_fixed_scalars": {"xi": 0.0},
                "warm_start_steps": 5000,
            },
        }

        fit_config = build_fit_config(variant, {"N": 4, "T": 3, "s": 1, "e": 2})
        self.assertEqual(
            OmegaConf.to_container(
                fit_config.estimation_params.warm_start_fixed_scalars, resolve=True
            ),
            {"xi": 0.0},
        )
        self.assertEqual(int(fit_config.estimation_params.warm_start_steps), 5000)

    def test_validate_fits_spec_allows_uv_ridge_for_concurrent_mode(self) -> None:
        spec_root = REPO_ROOT / "experiments" / f".tmp_spec_{uuid.uuid4().hex}"
        spec_root.mkdir(parents=True, exist_ok=True)
        spec_path = spec_root / "fits_spec.yaml"
        try:
            spec_path.write_text(
                "\n".join(
                    [
                        "base:",
                        "  fit_root_name: fits",
                        "  fit_manifest_path: tmp.csv",
                        "  optimizer:",
                        "    steps: 5",
                        "    tol: 1.0e-6",
                        "    seed: 0",
                        "  B: 1.0",
                        "  latent_rank: 1",
                        "  optimizer_mode: no_external_field",
                        "  lambda_nuclear: 0.0",
                        "  lambda_frobenius: 0.0",
                        "  lambda_uv_ridge: 0.0",
                        "  estimation:",
                        "    fixed_scalar_params: {}",
                        "variants:",
                        "  - name: concurrent_rank_1",
                        "    optimizer_mode: concurrent_latent_rank",
                        "    latent_rank: 1",
                        "    lambda_uv_ridge: 0.1",
                    ]
                ),
                encoding="utf-8",
            )
            validate_fits_spec(spec_path)
        finally:
            shutil.rmtree(spec_root, ignore_errors=True)

    def test_validate_fits_spec_rejects_nonpositive_v_column_l2_max(self) -> None:
        spec_root = REPO_ROOT / "experiments" / f".tmp_spec_{uuid.uuid4().hex}"
        spec_root.mkdir(parents=True, exist_ok=True)
        spec_path = spec_root / "fits_spec.yaml"
        try:
            spec_path.write_text(
                "\n".join(
                    [
                        "base:",
                        "  fit_root_name: fits",
                        "  fit_manifest_path: tmp.csv",
                        "  optimizer:",
                        "    steps: 5",
                        "    tol: 1.0e-6",
                        "    seed: 0",
                        "  B: 1.0",
                        "  latent_rank: 1",
                        "  optimizer_mode: alternating_latent_rank",
                        "  estimation:",
                        "    fixed_scalar_params: {}",
                        "variants:",
                        "  - name: alternating_rank_1_bad_v_constraint",
                        "    v_column_l2_max: 0.0",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "v_column_l2_max"):
                validate_fits_spec(spec_path)
        finally:
            shutil.rmtree(spec_root, ignore_errors=True)

    def test_build_fit_config_rejects_uv_ridge_for_unrelated_mode(self) -> None:
        variant = {
            "name": "bad_rank_0",
            "optimizer": {"steps": 5, "tol": 1.0e-6, "seed": 3},
            "optimizer_mode": "no_external_field",
            "latent_rank": 0,
            "lambda_uv_ridge": 0.25,
            "B": 1.0,
            "estimation": {"fixed_scalar_params": {}},
        }

        with self.assertRaisesRegex(ValueError, "lambda_uv_ridge"):
            build_fit_config(variant, {"N": 4, "T": 3, "s": 1})


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
        optimizer_mode: str = "no_external_field",
        B: float = 1.0,
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
                        "category": (
                            "metric"
                            if name
                            in {"final_loss", "field_rmse", "interaction_fro_error"}
                            else "scalar"
                        ),
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
            "optimizer_mode": optimizer_mode,
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
                    "  optimizer_mode: no_external_field",
                    "  lambda_frobenius: 0.0",
                    "  lambda_uv_ridge: 0.0",
                    "  estimation:",
                    "    fixed_scalar_params: {}",
                    "variants:",
                    "  - name: rank_0",
                    "  - name: rank_0_fixed_scalars",
                    "    estimation:",
                    "      fixed_scalar_params:",
                    "        beta: 0.2",
                    "        xi: 0.1",
                    "        eta: 0.05",
                    "  - name: nuclear_lambda_1e_2_B1",
                    "    optimizer_mode: nuclear_norm",
                    "    lambda_nuclear: 0.01",
                    "    B: 1.0",
                ]
            ),
            encoding="utf-8",
        )

        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        fit_manifest = run_fits(generation_manifest, fits_spec_path, overwrite=True)

        experiment_root = self.root / "generated" / "smoke_rank_0"
        fit_summary_csv = experiment_root / "fit_summary.csv"
        winners_csv = self.root / "generated" / "best_fit_by_experiment.csv"

        self.assertTrue(fit_summary_csv.exists())
        self.assertTrue(winners_csv.exists())

        with fit_summary_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(sum(row["is_best"] == "True" for row in rows), 1)
        self.assertEqual(len(rows), 3)
        self.assertIn("total_recovery_rmse", rows[0])
        self.assertIn("optimizer_mode", rows[0])
        self.assertIn("lambda_frobenius", rows[0])
        self.assertIn("lambda_uv_ridge", rows[0])
        self.assertIn("estimated_field_max_abs_entry", rows[0])
        self.assertIn("true_field_max_abs_entry", rows[0])
        self.assertNotIn("estimated_field_inf_norm", rows[0])
        self.assertNotIn("true_field_inf_norm", rows[0])
        nuclear_root = experiment_root / "fits" / "nuclear_lambda_1e_2_b1"
        self.assertTrue((nuclear_root / "mple_summary.csv").exists())
        self.assertTrue((nuclear_root / "estimated_field_artifacts.npz").exists())
        self.assertTrue((nuclear_root / "estimated_parameter_bundle.npz").exists())

        with winners_csv.open("r", encoding="utf-8", newline="") as handle:
            winner_rows = list(csv.DictReader(handle))
        self.assertEqual(len(winner_rows), 1)
        self.assertEqual(winner_rows[0]["experiment_name"], "smoke_rank_0")
        self.assertIn("total_recovery_rmse", winner_rows[0])
        self.assertIn("optimizer_mode", winner_rows[0])
        self.assertIn("lambda_frobenius", winner_rows[0])
        self.assertIn("lambda_uv_ridge", winner_rows[0])
        self.assertEqual(
            Path(fit_manifest), self.root / "generated" / "fit_manifest.csv"
        )


class PipelineStageRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "experiments" / f".tmp_stage_requests_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_generation_spec(self, experiments: list[dict[str, object] | str]) -> Path:
        normalized_experiments: list[dict[str, object]] = []
        for experiment in experiments:
            if isinstance(experiment, str):
                normalized_experiments.append({"name": experiment})
            else:
                normalized_experiments.append(dict(experiment))
        spec_path = self.root / "generation_spec.yaml"
        spec = {
            "base": {
                "experiment_root": f"{self.root.as_posix()}/generated",
                "manifest_path": f"{self.root.as_posix()}/generated/generation_manifest.csv",
                "dimensions": {"N": 6, "T": 4, "s": 1},
                "generation": {"gibbs_sweeps": 1, "seed": 7},
                "x0": {
                    "generator": "bernoulli",
                    "params": {"p": 0.5, "fixed_val": None},
                },
                "graph": {
                    "source": "generated",
                    "generator": "erdos_renyi",
                    "params": {"p": 0.5},
                    "artifact": {
                        "gamma_path": None,
                        "node_index_path": None,
                        "artifact_dir": None,
                        "network_name": None,
                        "trim_scope": None,
                    },
                },
                "intervention": {
                    "source": "generated",
                    "artifact": {
                        "panel_path": None,
                        "z0_path": None,
                        "artifact_dir": None,
                        "shared_panel_dir": None,
                        "outcome_code": None,
                        "intervention_code": None,
                        "lag_code": None,
                        "trim_scope": None,
                    },
                },
                "truth": {
                    "B": 1.0,
                    "latent_rank": 0,
                    "field_mode": "random_low_rank",
                    "field_params": {},
                    "scalars": {
                        "beta": 0.2,
                        "xi": 0.1,
                        "eta": 0.05,
                        "zeta": -0.1,
                        "psi": 0.2,
                    },
                },
            },
            "experiments": normalized_experiments,
        }
        OmegaConf.save(OmegaConf.create(spec), spec_path)
        return spec_path

    def _write_fit_spec(self, variants: list[dict[str, object] | str]) -> Path:
        normalized_variants: list[dict[str, object]] = []
        for variant in variants:
            if isinstance(variant, str):
                normalized_variants.append({"name": variant})
            else:
                normalized_variants.append(dict(variant))
        fits_spec_path = self.root / "fits_spec.yaml"
        spec = {
            "base": {
                "fit_root_name": "fits",
                "fit_manifest_path": f"{self.root.as_posix()}/generated/fit_manifest.csv",
                "optimizer": {"steps": 5, "tol": 1.0e-6, "seed": 0},
                "B": 1.0,
                "latent_rank": 0,
                "optimizer_mode": "no_external_field",
                "lambda_nuclear": 0.0,
                "lambda_frobenius": 0.0,
                "lambda_uv_ridge": 0.0,
                "estimation": {"fixed_scalar_params": {}},
            },
            "variants": normalized_variants,
        }
        OmegaConf.save(OmegaConf.create(spec), fits_spec_path)
        return fits_spec_path

    def _write_fake_sbatch(self) -> tuple[Path, Path, Path]:
        fake_sbatch_path = self.root / "fake_sbatch.sh"
        fake_counter_path = self.root / "fake_sbatch_counter.txt"
        fake_log_path = self.root / "fake_sbatch_log.txt"
        fake_sbatch_path.write_text(
            "\n".join(
                [
                    "#!/bin/bash",
                    "set -euo pipefail",
                    'count="0"',
                    'if [[ -f "${FAKE_SBATCH_COUNTER}" ]]; then',
                    '  count="$(cat "${FAKE_SBATCH_COUNTER}")"',
                    "fi",
                    'count="$((count + 1))"',
                    'printf "%s" "${count}" > "${FAKE_SBATCH_COUNTER}"',
                    'printf "%s|" "$#" >> "${FAKE_SBATCH_LOG}"',
                    'printf "<%s>" "$@" >> "${FAKE_SBATCH_LOG}"',
                    'printf "\\n" >> "${FAKE_SBATCH_LOG}"',
                    'printf "%s\\n" "job${count}"',
                ]
            ),
            encoding="utf-8",
        )
        fake_sbatch_path.chmod(0o755)
        return fake_sbatch_path, fake_counter_path, fake_log_path

    def test_write_generation_requests_writes_one_row_per_experiment(self) -> None:
        generation_spec_path = self._write_generation_spec(["exp_a", "exp_b"])

        request_path = write_generation_requests(generation_spec_path)

        rows = read_csv_manifest(request_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["experiment_slug"] for row in rows], ["exp_a", "exp_b"])
        self.assertTrue(request_path.name.endswith("generation_requests.csv"))

    def test_run_generation_request_only_materializes_targeted_experiment(self) -> None:
        generation_spec_path = self._write_generation_spec(["exp_a", "exp_b"])

        write_generation_requests(generation_spec_path)
        run_generation_request(generation_spec_path, "exp_a", overwrite=True)

        self.assertTrue((self.root / "generated" / "exp_a" / "panel_data.npz").exists())
        self.assertFalse((self.root / "generated" / "exp_b").exists())

    def test_refresh_generation_manifest_rebuilds_manifest_from_outputs(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [
                {
                    "name": "exp_rank_2",
                    "truth": {
                        "latent_rank": 2,
                        "field_mode": "random_low_rank",
                    },
                }
            ]
        )

        write_generation_requests(generation_spec_path)
        run_generation_request(generation_spec_path, "exp_rank_2", overwrite=True)
        manifest_path = refresh_generation_manifest(generation_spec_path)

        rows = read_csv_manifest(manifest_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["experiment_name"], "exp_rank_2")
        self.assertEqual(rows[0]["field_mode"], "random_low_rank")
        with np.load(
            self.root / "generated" / "exp_rank_2" / "field_artifacts.npz",
            allow_pickle=False,
        ) as data:
            expected_latent_rank = int(np.asarray(data["latent_rank"]).item())
        self.assertEqual(int(rows[0]["latent_rank"]), expected_latent_rank)

    def test_write_fit_requests_writes_cartesian_product(self) -> None:
        generation_spec_path = self._write_generation_spec(["exp_a", "exp_b"])
        fits_spec_path = self._write_fit_spec(["rank_0", "rank_0_fixed"])
        generation_manifest = run_generation(generation_spec_path, overwrite=True)

        request_path = write_fit_requests(generation_manifest, fits_spec_path)

        rows = read_csv_manifest(request_path)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {(row["experiment_slug"], row["variant_slug"]) for row in rows},
            {
                ("exp_a", "rank_0"),
                ("exp_a", "rank_0_fixed"),
                ("exp_b", "rank_0"),
                ("exp_b", "rank_0_fixed"),
            },
        )

    def test_run_fit_request_writes_one_fit_folder(self) -> None:
        generation_spec_path = self._write_generation_spec(["exp_a"])
        fits_spec_path = self._write_fit_spec(
            [
                "rank_0",
                {
                    "name": "nuclear_lambda_1e_2",
                    "optimizer_mode": "nuclear_norm",
                    "lambda_nuclear": 0.01,
                },
            ]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)

        write_fit_requests(generation_manifest, fits_spec_path)
        run_fit_request(
            generation_manifest,
            fits_spec_path,
            "exp_a",
            "rank_0",
            overwrite=True,
        )

        rank_0_root = self.root / "generated" / "exp_a" / "fits" / "rank_0"
        nuclear_root = self.root / "generated" / "exp_a" / "fits" / "nuclear_lambda_1e_2"
        self.assertTrue((rank_0_root / "mple_summary.csv").exists())
        self.assertFalse(nuclear_root.exists())

    def test_refresh_fit_manifest_rebuilds_manifest_and_reports(self) -> None:
        generation_spec_path = self._write_generation_spec(["exp_a"])
        fits_spec_path = self._write_fit_spec(["rank_0"])
        generation_manifest = run_generation(generation_spec_path, overwrite=True)

        write_fit_requests(generation_manifest, fits_spec_path)
        run_fit_request(
            generation_manifest,
            fits_spec_path,
            "exp_a",
            "rank_0",
            overwrite=True,
        )
        fit_manifest = refresh_fit_manifest(generation_manifest, fits_spec_path)

        rows = read_csv_manifest(fit_manifest)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["experiment_name"], "exp_a")
        self.assertEqual(rows[0]["variant_name"], "rank_0")
        self.assertTrue(
            (self.root / "generated" / "exp_a" / "fit_summary.csv").exists()
        )
        self.assertTrue(
            (self.root / "generated" / "best_fit_by_experiment.csv").exists()
        )

    @unittest.skipIf(shutil.which("bash") is None, "bash is required for shell submission test")
    def test_submit_generation_jobs_submits_workers_and_report(self) -> None:
        bash_path = shutil.which("bash")
        if bash_path is None or "system32" in bash_path.lower():
            self.skipTest("portable bash is not available in this environment")
        generation_spec_path = self._write_generation_spec(["exp_a", "exp_b"])
        fake_sbatch_path, fake_counter_path, fake_log_path = self._write_fake_sbatch()

        result = subprocess.run(
            [bash_path, "submit_generation_jobs.sh"],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "GENERATION_SPEC_PATH": str(generation_spec_path),
                "SBATCH_BIN": str(fake_sbatch_path),
                "WORKER_SCRIPT": "run_generation_job.sh",
                "FAKE_SBATCH_COUNTER": str(fake_counter_path),
                "FAKE_SBATCH_LOG": str(fake_log_path),
            },
        )

        log_lines = fake_log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log_lines), 3)
        self.assertIn("run_generation_job.sh exp_a", log_lines[0])
        self.assertIn("run_generation_job.sh exp_b", log_lines[1])
        self.assertIn("--refresh_manifest", log_lines[2])
        self.assertEqual(result.stdout.strip(), "job3")

    @unittest.skipIf(shutil.which("bash") is None, "bash is required for shell submission test")
    def test_submit_fit_jobs_submits_workers_and_report(self) -> None:
        bash_path = shutil.which("bash")
        if bash_path is None or "system32" in bash_path.lower():
            self.skipTest("portable bash is not available in this environment")
        generation_manifest_path = self.root / "generation_manifest.csv"
        experiment_root = self.root / "generated" / "exp_a"
        experiment_root.mkdir(parents=True, exist_ok=True)
        with generation_manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "experiment_name",
                    "experiment_slug",
                    "descriptor",
                    "experiment_path",
                    "intervention_source",
                    "graph_source",
                    "N",
                    "T",
                    "s",
                    "has_truth",
                    "field_mode",
                    "latent_rank",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "experiment_name": "exp_a",
                    "experiment_slug": "exp_a",
                    "descriptor": "exp_a",
                    "experiment_path": str(experiment_root.resolve()),
                    "intervention_source": "generated",
                    "graph_source": "generated",
                    "N": 6,
                    "T": 4,
                    "s": 1,
                    "has_truth": True,
                    "field_mode": "random_low_rank",
                    "latent_rank": 0,
                }
            )
        fits_spec_path = self._write_fit_spec(["rank_0", "rank_0_fixed"])
        fake_sbatch_path, fake_counter_path, fake_log_path = self._write_fake_sbatch()

        result = subprocess.run(
            [bash_path, "submit_fit_jobs.sh"],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "GENERATION_MANIFEST_PATH": str(generation_manifest_path),
                "FITS_SPEC_PATH": str(fits_spec_path),
                "SBATCH_BIN": str(fake_sbatch_path),
                "WORKER_SCRIPT": "run_fit_job.sh",
                "FAKE_SBATCH_COUNTER": str(fake_counter_path),
                "FAKE_SBATCH_LOG": str(fake_log_path),
            },
        )

        log_lines = fake_log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log_lines), 3)
        self.assertIn("run_fit_job.sh exp_a rank_0", log_lines[0])
        self.assertIn("run_fit_job.sh exp_a rank_0_fixed", log_lines[1])
        self.assertIn("--refresh_manifest", log_lines[2])
        self.assertEqual(result.stdout.strip(), "job3")

    @unittest.skipIf(shutil.which("bash") is None, "bash is required for shell orchestration test")
    def test_run_tests_sh_waits_between_stage_submissions(self) -> None:
        bash_path = shutil.which("bash")
        if bash_path is None or "system32" in bash_path.lower():
            self.skipTest("portable bash is not available in this environment")
        log_path = self.root / "orchestration.log"
        generation_submitter = self.root / "fake_generation_submitter.sh"
        fit_submitter = self.root / "fake_fit_submitter.sh"
        posterior_submitter = self.root / "fake_posterior_submitter.sh"
        intervention_script = self.root / "fake_intervention.sh"
        sacct_script = self.root / "fake_sacct.sh"
        squeue_script = self.root / "fake_squeue.sh"

        generation_submitter.write_text(
            "#!/bin/bash\nset -euo pipefail\nprintf 'generation_submit\\n' >> \"${STAGE_LOG}\"\nprintf 'job-generation\\n'\n",
            encoding="utf-8",
        )
        fit_submitter.write_text(
            "#!/bin/bash\nset -euo pipefail\nprintf 'fit_submit\\n' >> \"${STAGE_LOG}\"\nprintf 'job-fit\\n'\n",
            encoding="utf-8",
        )
        posterior_submitter.write_text(
            "#!/bin/bash\nset -euo pipefail\nprintf 'posterior_submit\\n' >> \"${STAGE_LOG}\"\nprintf 'job-posterior\\n'\n",
            encoding="utf-8",
        )
        intervention_script.write_text(
            "#!/bin/bash\nset -euo pipefail\nprintf 'intervention\\n' >> \"${STAGE_LOG}\"\n",
            encoding="utf-8",
        )
        sacct_script.write_text(
            "#!/bin/bash\nset -euo pipefail\njob_id=\"\"\nwhile [[ $# -gt 0 ]]; do\n  if [[ \"$1\" == \"-j\" ]]; then\n    job_id=\"$2\"\n    shift 2\n    continue\n  fi\n  shift\n done\nprintf 'wait:%s\\n' \"${job_id}\" >> \"${STAGE_LOG}\"\nprintf 'COMPLETED\\n'\n",
            encoding="utf-8",
        )
        squeue_script.write_text(
            "#!/bin/bash\nset -euo pipefail\n",
            encoding="utf-8",
        )
        for script_path in [
            generation_submitter,
            fit_submitter,
            posterior_submitter,
            intervention_script,
            sacct_script,
            squeue_script,
        ]:
            script_path.chmod(0o755)

        subprocess.run(
            [bash_path, "run_tests.sh"],
            check=True,
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "STAGE_LOG": str(log_path),
                "GEN_MANIFEST": str(self.root / "generation_manifest.csv"),
                "FIT_MANIFEST": str(self.root / "fit_manifest.csv"),
                "GENERATION_SUBMITTER": str(generation_submitter),
                "FIT_SUBMITTER": str(fit_submitter),
                "POSTERIOR_PREDICTIVE_SUBMITTER": str(posterior_submitter),
                "INTERVENTION_LIBRARY_SCRIPT": str(intervention_script),
                "SACCT_BIN": str(sacct_script),
                "SQUEUE_BIN": str(squeue_script),
                "SLEEP_BIN": str(squeue_script),
                "WAIT_POLL_SECONDS": "0",
            },
        )

        self.assertEqual(
            log_path.read_text(encoding="utf-8").splitlines(),
            [
                "generation_submit",
                "wait:job-generation",
                "fit_submit",
                "wait:job-fit",
                "intervention",
                "posterior_submit",
                "wait:job-posterior",
            ],
        )


class USCountyVaccinationSharedPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / f".tmp_uc_{uuid.uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(io_path(self.root), ignore_errors=True)

    def _write_target_pairs(self, rows: list[dict[str, object]]) -> Path:
        target_pairs_path = self.root / "target_pairs.csv"
        with target_pairs_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "experiment_name",
                    "source_type",
                    "variant_name",
                    "intervention_source",
                    "intervention_name",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return target_pairs_path

    def _write_us_county_experiment(self) -> tuple[Path, Path]:
        experiment_name = (
            "outcome_death_rate_100k_ge_2__intervention_complete_cov_ge_20"
            "__lag_2w__contiguity"
        )
        experiment_root = self.root / experiment_name
        x = np.array(
            [
                [-1, 1, -1, 1],
                [1, 1, -1, -1],
                [1, -1, 1, -1],
                [-1, -1, 1, 1],
            ],
            dtype=np.int8,
        )
        z = np.array(
            [
                [-1, -1, -1, -1],
                [1, -1, 1, -1],
                [1, 1, 1, -1],
                [1, 1, 1, 1],
            ],
            dtype=np.int8,
        )
        x_0 = np.array([-1, -1, 1, 1], dtype=np.int8)
        z_0 = np.array([-1, -1, -1, -1], dtype=np.int8)
        node_table = pd.DataFrame(
            {
                "fips": ["01001", "01003", "01005", "01007"],
                "node_index": [0, 1, 2, 3],
                "county": ["a", "b", "c", "d"],
                "state_name": ["Alabama"] * 4,
            }
        )
        time_index = pd.DataFrame(
            {
                "WeekStartDate": pd.date_range("2021-01-03", periods=5, freq="W-SUN"),
                "WeekEndDate": pd.date_range("2021-01-09", periods=5, freq="W-SAT"),
                "iso_year": [2021] * 5,
                "iso_week": [1, 2, 3, 4, 5],
                "model_index": [0, 1, 2, 3, 4],
            }
        )
        x_all = np.vstack([x_0[None, :], x])
        z_all = np.vstack([z_0[None, :], z])
        panel = pd.DataFrame(
            {
                "WeekEndDate": np.repeat(time_index["WeekEndDate"].to_numpy(), 4),
                "fips": np.tile(node_table["fips"].to_numpy(), 5),
                "Outcome_pm1": x_all.reshape(-1),
                "Intervention_pm1": z_all.reshape(-1),
            }
        )
        gamma = sparse.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
            / 2.0
        )
        config = create_us_county_config(
            n_nodes=4,
            t_steps=4,
            s=1,
            outcome_code="death_rate_100k_ge_2",
            intervention_code="complete_cov_ge_20",
            lag_code="2w",
            network_name="contiguity",
            state_scope_label="Mainland US counties with total_population >= 2000",
        )
        metadata = {
            "source": "USCountyVaccination",
            "has_truth": False,
            "outcome_code": "death_rate_100k_ge_2",
            "intervention_code": "complete_cov_ge_20",
            "lag_code": "2w",
            "network_name": "contiguity",
            "trim_applied": True,
            "node_count": 4,
            "time_steps": 4,
            "pre_intervention_steps": 1,
        }
        save_us_county_experiment(
            experiment_root,
            config,
            metadata,
            gamma,
            pd.DataFrame(
                {"fips": ["01001", "01003"], "neighbor_fips": ["01003", "01005"]}
            ),
            panel,
            node_table,
            time_index,
            x,
            z,
            x_0,
            z_0,
        )
        manifest_path = self.root / "generation_manifest.csv"
        row = {
            "experiment_name": experiment_name,
            "experiment_slug": experiment_name,
            "descriptor": experiment_name,
            "experiment_path": str(experiment_root.resolve()),
            "intervention_source": "real_data",
            "graph_source": "contiguity",
            "N": 4,
            "T": 4,
            "s": 1,
            "has_truth": False,
            "outcome_code": "death_rate_100k_ge_2",
            "intervention_code": "complete_cov_ge_20",
            "lag_code": "2w",
            "network_name": "contiguity",
        }
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return experiment_root, manifest_path

    def _write_us_county_realized_artifacts(self, output_root: Path) -> dict[str, object]:
        node_order = ["01001", "01003", "01005", "01007"]
        features = pd.DataFrame(
            {
                "fips": node_order,
                "county": ["a", "b", "c", "d"],
                "state_name": ["Alabama"] * 4,
                "STATEFP": ["01"] * 4,
                "feature_basis_mode": ["zero"] * 4,
                "total_population": [10000, 12000, 14000, 16000],
            }
        )
        centroids = pd.DataFrame(
            {
                "fips": node_order,
                "county": ["a", "b", "c", "d"],
                "state_name": ["Alabama"] * 4,
                "longitude": [-86.0, -86.5, -87.0, -87.5],
                "latitude": [32.5, 32.6, 32.7, 32.8],
            }
        )
        node_table = pd.DataFrame(
            {
                "fips": node_order,
                "node_index": [0, 1, 2, 3],
                "county": ["a", "b", "c", "d"],
                "state_name": ["Alabama"] * 4,
            }
        )
        time_index = pd.DataFrame(
            {
                "WeekStartDate": pd.date_range("2021-01-03", periods=5, freq="W-SUN"),
                "WeekEndDate": pd.date_range("2021-01-09", periods=5, freq="W-SAT"),
                "iso_year": [2021] * 5,
                "iso_week": [1, 2, 3, 4, 5],
                "model_index": [0, 1, 2, 3, 4],
            }
        )
        x = np.array(
            [
                [-1, 1, -1, 1],
                [1, 1, -1, -1],
                [1, -1, 1, -1],
                [-1, -1, 1, 1],
            ],
            dtype=np.int8,
        )
        z = np.array(
            [
                [-1, -1, -1, -1],
                [1, -1, 1, -1],
                [1, 1, 1, -1],
                [1, 1, 1, 1],
            ],
            dtype=np.int8,
        )
        x_0 = np.array([-1, -1, 1, 1], dtype=np.int8)
        z_0 = np.array([-1, -1, -1, -1], dtype=np.int8)
        gamma = sparse.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
            / 2.0
        )
        adjacency_edges = pd.DataFrame(
            {
                "fips": ["01001", "01003", "01003", "01005", "01005", "01007"],
                "neighbor_fips": ["01003", "01001", "01005", "01003", "01007", "01005"],
            }
        )
        panel = assembled_panel_from_arrays(
            x=x,
            z=z,
            x_0=x_0,
            z_0=z_0,
            time_index=time_index,
            node_order=node_order,
            outcome_code="death_rate_100k_ge_2",
            intervention_code="complete_cov_ge_20",
        )
        trim_label = "mainland_us_and_total_population_ge_2000"
        state_label = "Mainland US counties with total_population >= 2000"
        write_realized_binary_artifact(
            output_root
            / "realized_outcomes"
            / realized_outcome_name("death_rate_100k_ge_2", True),
            RealizedBinaryArtifact(
                code="death_rate_100k_ge_2",
                panel_key="x",
                values=x,
                initial_values=x_0,
                observed_mask=np.ones_like(x, dtype=bool),
                initial_observed_mask=np.ones_like(x_0, dtype=bool),
                node_order=node_order,
                time_index=time_index,
                artifact_dir=output_root
                / "realized_outcomes"
                / realized_outcome_name("death_rate_100k_ge_2", True),
                metadata={
                    "outcome_code": "death_rate_100k_ge_2",
                    "trim_applied": True,
                    "trim_rule": trim_label,
                },
            ),
        )
        write_realized_binary_artifact(
            output_root
            / "realized_interventions"
            / realized_intervention_name("complete_cov_ge_20", "2w", True),
            RealizedBinaryArtifact(
                code="complete_cov_ge_20",
                panel_key="z",
                values=z,
                initial_values=z_0,
                observed_mask=np.ones_like(z, dtype=bool),
                initial_observed_mask=np.ones_like(z_0, dtype=bool),
                node_order=node_order,
                time_index=time_index,
                artifact_dir=output_root
                / "realized_interventions"
                / realized_intervention_name("complete_cov_ge_20", "2w", True),
                metadata={
                    "intervention_code": "complete_cov_ge_20",
                    "lag_code": "2w",
                    "trim_applied": True,
                    "trim_rule": trim_label,
                },
            ),
        )
        write_realized_network_artifact(
            output_root
            / "realized_networks"
            / realized_network_name("contiguity", True),
            RealizedNetworkArtifact(
                network_name="contiguity",
                gamma_matrix=gamma,
                adjacency_edges=adjacency_edges,
                node_order=node_order,
                artifact_dir=output_root
                / "realized_networks"
                / realized_network_name("contiguity", True),
                metadata={"network_name": "contiguity", "trim_applied": True},
            ),
        )
        write_shared_panel_artifacts(
            output_root
            / "shared_panels"
            / shared_panel_name(
                "death_rate_100k_ge_2",
                "complete_cov_ge_20",
                "2w",
                True,
            ),
            panel=panel,
            node_table=node_table,
            time_index=time_index,
            x=x,
            z=z,
            x_0=x_0,
            z_0=z_0,
            metadata={
                "source": "USCountyVaccination",
                "state": state_label,
                "outcome_code": "death_rate_100k_ge_2",
                "intervention_code": "complete_cov_ge_20",
                "lag_code": "2w",
                "trim_applied": True,
                "trim_rule": trim_label,
                "requested_node_count": 4,
                "dropped_node_count": 0,
                "requested_calendar_weeks": 5,
                "realized_calendar_weeks": 5,
                "weeks_dropped_due_to_missing_or_lag": 0,
                "support_selection_rule": "max_complete_suffix_by_node_week_area",
                "requested_start_date": "2021-01-09",
                "realized_week_start_date": "2021-01-03",
                "realized_week_end_date": "2021-02-06",
                "time_steps": 4,
                "pre_intervention_steps": 1,
            },
        )
        return {
            "features": features,
            "centroids": centroids,
            "x": x,
            "z": z,
            "x_0": x_0,
            "z_0": z_0,
            "time_index": time_index,
            "base_experiment_name": (
                "outcome_death_rate_100k_ge_2__intervention_complete_cov_ge_20"
                "__lag_2w__contiguity"
            ),
        }

    def _materializer_args(
        self,
        output_root: Path,
        *,
        start_dates: list[str] | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            overwrite=True,
            max_experiments=None,
            lags=["2w"],
            outcomes=["death_rate_100k_ge_2"],
            interventions=["complete_cov_ge_20"],
            networks=["contiguity"],
            output_root=output_root,
            start_dates=start_dates,
            trim=True,
        )

    def _run_us_county_materializer(
        self,
        *,
        output_root: Path,
        start_dates: list[str] | None = None,
    ) -> dict[str, object]:
        fixture = self._write_us_county_realized_artifacts(output_root)
        args = self._materializer_args(output_root, start_dates=start_dates)
        with mock.patch.object(
            uscounty_materializer,
            "load_inputs",
            return_value=(pd.DataFrame(), fixture["features"], fixture["centroids"]),
        ):
            uscounty_materializer.create_experiment_folders(args)
        return fixture

    def test_us_county_experiment_root_matches_shared_artifact_contract(self) -> None:
        experiment_root, manifest_path = self._write_us_county_experiment()

        dims = infer_panel_dimensions(experiment_root)
        panel_context = load_experiment_panel_context(experiment_root)
        artifacts = load_model_artifacts(experiment_root)
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            manifest_row = next(csv.DictReader(handle))

        self.assertEqual(dims, {"N": 4, "T": 4, "s": 1})
        self.assertEqual(panel_context["x"].shape, (4, 4))
        self.assertEqual(artifacts.field_matrix.shape, (4, 4))
        self.assertTrue((experiment_root / "node_index.csv").exists())
        self.assertTrue((experiment_root / "time_index.csv").exists())
        for key in [
            "experiment_name",
            "experiment_path",
            "intervention_source",
            "graph_source",
            "N",
            "T",
            "s",
            "has_truth",
        ]:
            self.assertIn(key, manifest_row)

    def test_us_county_start_index_exact_match(self) -> None:
        fixture = self._write_us_county_realized_artifacts(self.root)
        time_index = fixture["time_index"]
        start_index, resolved = uscounty_materializer._resolve_start_index(
            time_index, "2021-01-23"
        )
        self.assertEqual(start_index, 2)
        self.assertEqual(resolved, "2021-01-23")

    def test_us_county_start_index_rounds_forward(self) -> None:
        fixture = self._write_us_county_realized_artifacts(self.root)
        time_index = fixture["time_index"]
        start_index, resolved = uscounty_materializer._resolve_start_index(
            time_index, "2021-01-18"
        )
        self.assertEqual(start_index, 2)
        self.assertEqual(resolved, "2021-01-23")

    def test_us_county_start_index_rejects_after_last_week(self) -> None:
        fixture = self._write_us_county_realized_artifacts(self.root)
        time_index = fixture["time_index"]
        with self.assertRaisesRegex(ValueError, "after the last available week"):
            uscounty_materializer._resolve_start_index(time_index, "2021-02-20")

    def test_us_county_start_index_rejects_no_transition_week(self) -> None:
        fixture = self._write_us_county_realized_artifacts(self.root)
        time_index = fixture["time_index"]
        with self.assertRaisesRegex(ValueError, "leaves no transition weeks to fit"):
            uscounty_materializer._resolve_start_index(time_index, "2021-02-06")

    def test_us_county_materializer_preserves_unsliced_behavior(self) -> None:
        fixture = self._run_us_county_materializer(output_root=self.root / "generated")
        manifest_path = self.root / "generated" / "generation_manifest.csv"
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["experiment_name"], fixture["base_experiment_name"])
        self.assertEqual(row["T"], "4")
        self.assertEqual(row["s"], "1")
        self.assertNotIn("requested_start_date", row)
        self.assertNotIn("resolved_start_week_end_date", row)
        for key in [
            "field_mode",
            "field_basis_mode",
            "field_basis_names",
            "model_field_mode",
            "latent_rank",
            "latent_B",
            "tau_zero_mean",
            "tau_smoothness_lambda",
        ]:
            self.assertNotIn(key, row)

    def test_us_county_materializer_drops_legacy_field_and_tau_settings(self) -> None:
        fixture = self._run_us_county_materializer(output_root=self.root / "generated")
        experiment_root = self.root / "generated" / fixture["base_experiment_name"]

        config = OmegaConf.to_container(
            OmegaConf.load(experiment_root / "realized_config.yaml"),
            resolve=True,
        )
        metadata = OmegaConf.to_container(
            OmegaConf.load(experiment_root / "experiment_metadata.yaml"),
            resolve=True,
        )

        self.assertEqual(
            config["global_params"],
            {
                "N": 4,
                "T": 4,
                "s": 1,
                "gamma_matrix_generator": "real_data",
                "x_0_generator": "observed",
            },
        )
        self.assertEqual(
            config["estimation_params"],
            {
                "beta": 0.0,
                "eta": 0.0,
                "tau_params": None,
                "fixed_scalar_params": {},
            },
        )
        self.assertEqual(
            config["real_data_params"],
            {
                "source": "USCountyVaccination",
                "state": "Mainland US counties with total_population >= 2000",
                "outcome_code": "death_rate_100k_ge_2",
                "intervention_code": "complete_cov_ge_20",
                "lag_code": "2w",
                "lag_application": "intervention_only",
                "network_name": "contiguity",
            },
        )
        for key in [
            "B",
            "latent_rank",
            "basis_params",
            "field_basis_mode",
            "field_basis_names",
            "model_field_mode",
            "latent_B",
            "tau_zero_mean",
            "tau_smoothness_lambda",
        ]:
            self.assertNotIn(key, config["global_params"])
            self.assertNotIn(key, config["estimation_params"])
            self.assertNotIn(key, config["real_data_params"])
            self.assertNotIn(key, metadata)

    def test_us_county_materializer_rejects_removed_legacy_flags(self) -> None:
        removed_flags = [
            "--field_mode",
            "--latent_rank",
            "--latent_B",
            "--tau_zero_mean",
            "--tau_smoothness_lambda",
        ]
        for removed_flag in removed_flags:
            with self.subTest(flag=removed_flag):
                argv = ["create_us_county_vaccination_experiments.py", removed_flag]
                if removed_flag != "--tau_zero_mean":
                    argv.append("value")
                with mock.patch.object(sys, "argv", argv):
                    with self.assertRaises(SystemExit):
                        uscounty_materializer.parse_args()

    def test_us_county_materializer_slices_one_start_date(self) -> None:
        fixture = self._run_us_county_materializer(
            output_root=self.root / "generated",
            start_dates=["2021-01-23"],
        )
        manifest_path = self.root / "generated" / "generation_manifest.csv"
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(
            row["experiment_name"],
            fixture["base_experiment_name"] + "__start_2021_01_23",
        )
        self.assertEqual(row["T"], "2")
        self.assertEqual(row["s"], "0")
        self.assertEqual(row["requested_start_date"], "2021-01-23")
        self.assertEqual(row["resolved_start_week_end_date"], "2021-01-23")
        self.assertEqual(row["start_index"], "2")
        self.assertEqual(row["dropped_transition_weeks_for_start"], "2")

        derived_root = Path(row["experiment_path"])
        self.assertTrue((derived_root / "panel_data.npz").exists())
        self.assertTrue((derived_root / "x_0.npy").exists())
        self.assertTrue((derived_root / "z_0.npy").exists())
        self.assertTrue((derived_root / "time_index.csv").exists())
        self.assertTrue(
            np.array_equal(
                np.load(derived_root / "x_0.npy"),
                np.array([1, 1, -1, -1], dtype=np.int8),
            )
        )
        self.assertTrue(
            np.array_equal(
                np.load(derived_root / "z_0.npy"),
                np.array([1, -1, 1, -1], dtype=np.int8),
            )
        )
        derived_dims = infer_panel_dimensions(derived_root)
        self.assertEqual(derived_dims, {"N": 4, "T": 2, "s": 0})

        metadata = OmegaConf.to_container(
            OmegaConf.load(derived_root / "experiment_metadata.yaml"),
            resolve=True,
        )
        self.assertEqual(metadata["requested_start_date"], "2021-01-23")
        self.assertEqual(metadata["resolved_start_week_end_date"], "2021-01-23")
        self.assertEqual(int(metadata["start_index"]), 2)
        self.assertEqual(int(metadata["dropped_transition_weeks_for_start"]), 2)

    def test_us_county_materializer_slices_multiple_start_dates(self) -> None:
        fixture = self._run_us_county_materializer(
            output_root=self.root / "generated",
            start_dates=["2021-01-16", "2021-01-23"],
        )
        manifest_path = self.root / "generated" / "generation_manifest.csv"
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        names = {row["experiment_name"] for row in rows}
        self.assertEqual(
            names,
            {
                fixture["base_experiment_name"] + "__start_2021_01_16",
                fixture["base_experiment_name"] + "__start_2021_01_23",
            },
        )

    def test_us_county_sliced_experiment_roots_work_with_fit_pipeline(self) -> None:
        fixture = self._run_us_county_materializer(
            output_root=self.root / "generated",
            start_dates=["2021-01-23"],
        )
        manifest_path = self.root / "generated" / "generation_manifest.csv"
        fits_spec_path = self.root / "fits_spec.yaml"
        fits_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    "  fit_root_name: fits",
                    f"  fit_manifest_path: {self.root.as_posix()}/fit_manifest.csv",
                    "  optimizer:",
                    "    steps: 5",
                    "    tol: 1.0e-6",
                    "    seed: 0",
                    "  B: 1.0",
                    "  latent_rank: 0",
                    "  estimation:",
                    "    fixed_scalar_params: {}",
                    "variants:",
                    "  - name: rank_0",
                ]
            ),
            encoding="utf-8",
        )

        fit_manifest = run_fits(manifest_path, fits_spec_path, overwrite=True)
        fit_rows = read_csv_manifest(fit_manifest)
        self.assertEqual(len(fit_rows), 1)
        self.assertEqual(
            fit_rows[0]["experiment_name"],
            fixture["base_experiment_name"] + "__start_2021_01_23",
        )

    def test_us_county_truth_targets_are_rejected(self) -> None:
        experiment_root, _ = self._write_us_county_experiment()
        target_pairs_path = self._write_target_pairs(
            [
                {
                    "experiment_name": experiment_root.name,
                    "source_type": "truth",
                    "variant_name": "",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "has_truth=false"):
            resolve_target_pairs(
                target_pairs_path,
                {
                    experiment_root.name: {
                        "experiment_name": experiment_root.name,
                        "experiment_path": str(experiment_root),
                    }
                },
                {},
            )

    def test_us_county_fit_and_counterfactual_pipeline_smoke(self) -> None:
        experiment_root, generation_manifest = self._write_us_county_experiment()
        fits_spec_path = self.root / "fits_spec.yaml"
        intervention_spec_path = self.root / "intervention_library_spec.yaml"
        predictive_spec_path = self.root / "posterior_predictive_spec.yaml"
        target_pairs_path = self.root / "target_pairs.csv"
        fits_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    "  fit_root_name: fits",
                    f"  fit_manifest_path: {self.root.as_posix()}/fit_manifest.csv",
                    "  optimizer:",
                    "    steps: 5",
                    "    tol: 1.0e-6",
                    "    seed: 0",
                    "  B: 1.0",
                    "  latent_rank: 0",
                    "  estimation:",
                    "    fixed_scalar_params: {}",
                    "variants:",
                    "  - name: rank_0",
                ]
            ),
            encoding="utf-8",
        )
        intervention_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    f"  experiment_name: {experiment_root.name}",
                    "interventions:",
                    "  - name: all_ones_from_s",
                    "    source_kind: full_on",
                    "    activation_scope: from_s",
                ]
            ),
            encoding="utf-8",
        )
        predictive_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    "  num_samples: 2",
                    "  gibbs_sweeps: 1",
                    "  seed: 0",
                    "runs:",
                    "  - name: default",
                ]
            ),
            encoding="utf-8",
        )
        target_pairs_path.write_text(
            "\n".join(
                [
                    "experiment_name,source_type,variant_name,intervention_source,intervention_name",
                    f"{experiment_root.name},fit,rank_0,saved_intervention,all_ones_from_s",
                ]
            ),
            encoding="utf-8",
        )

        fit_manifest = run_fits(generation_manifest, fits_spec_path, overwrite=True)
        run_intervention_library(
            generation_manifest,
            intervention_spec_path,
            overwrite=True,
        )
        row = run_posterior_predictive(
            generation_manifest,
            fit_manifest,
            target_pairs_path,
            predictive_spec_path,
            experiment_name=experiment_root.name,
            source_type="fit",
            variant_name="rank_0",
            intervention_source="saved_intervention",
            intervention_name="all_ones_from_s",
            run_name="default",
            overwrite=True,
        )

        counterfactual_root = (
            experiment_root
            / "counterfactual"
            / "fit_rank_0"
            / "all_ones_from_s"
            / "default"
        )
        self.assertEqual(
            Path(str(row["output_path"])).resolve(),
            counterfactual_root.resolve(),
        )
        self.assertTrue(
            Path(io_path(counterfactual_root / "counterfactual_summary.csv")).exists()
        )
        self.assertTrue(
            Path(
                io_path(counterfactual_root / "counterfactual_unit_summary.csv")
            ).exists()
        )


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
                fieldnames=[
                    "experiment_name",
                    "source_type",
                    "variant_name",
                    "intervention_source",
                    "intervention_name",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return target_pairs_path

    def _write_counterfactual_summary_outputs(
        self,
        output_root: Path,
        *,
        overall_mean: float | None,
        overall_q025: float | None,
        overall_q500: float | None,
        overall_q975: float | None,
        post_mean: float | None,
        post_q025: float | None,
        post_q500: float | None,
        post_q975: float | None,
        unit_means: list[float],
        unit_q025: list[float],
        unit_q975: list[float],
    ) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_rows = [
            {
                "statistic": "overall_mean_magnetization",
                "sample_mean": overall_mean,
                "sample_std": 0.0 if overall_mean is not None else "",
                "q025": overall_q025,
                "q500": overall_q500,
                "q975": overall_q975,
                "num_finite_samples": 4 if overall_mean is not None else 0,
            },
            {
                "statistic": "post_intervention_mean_magnetization",
                "sample_mean": post_mean,
                "sample_std": 0.0 if post_mean is not None else "",
                "q025": post_q025,
                "q500": post_q500,
                "q975": post_q975,
                "num_finite_samples": 4 if post_mean is not None else 0,
            },
        ]
        with (output_root / "counterfactual_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "statistic",
                    "sample_mean",
                    "sample_std",
                    "q025",
                    "q500",
                    "q975",
                    "num_finite_samples",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        unit_rows = [
            {
                "unit_index": unit_index,
                "sample_mean": mean,
                "sample_std": 0.0,
                "q025": q025,
                "q500": mean,
                "q975": q975,
                "num_finite_samples": 4,
            }
            for unit_index, (mean, q025, q975) in enumerate(
                zip(unit_means, unit_q025, unit_q975)
            )
        ]
        with (output_root / "counterfactual_unit_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "unit_index",
                    "sample_mean",
                    "sample_std",
                    "q025",
                    "q500",
                    "q975",
                    "num_finite_samples",
                ],
            )
            writer.writeheader()
            writer.writerows(unit_rows)

    def _write_predictive_stats_output(
        self,
        output_root: Path,
        *,
        overall_mean: float,
        post_mean: float,
    ) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        with (output_root / "posterior_predictive_stats.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "statistic",
                    "observed_value",
                    "sample_mean",
                    "sample_std",
                    "z_score",
                    "tail_probability",
                    "q025",
                    "q500",
                    "q975",
                    "in_95_interval",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "statistic": "overall_mean_magnetization",
                        "observed_value": overall_mean,
                        "sample_mean": overall_mean,
                        "sample_std": 0.0,
                        "z_score": 0.0,
                        "tail_probability": 1.0,
                        "q025": overall_mean,
                        "q500": overall_mean,
                        "q975": overall_mean,
                        "in_95_interval": True,
                    },
                    {
                        "statistic": "post_intervention_mean_magnetization",
                        "observed_value": post_mean,
                        "sample_mean": post_mean,
                        "sample_std": 0.0,
                        "z_score": 0.0,
                        "tail_probability": 1.0,
                        "q025": post_mean,
                        "q500": post_mean,
                        "q975": post_mean,
                        "in_95_interval": True,
                    },
                ]
            )

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

    def test_load_fit_parameter_bundle_propagates_beta_mask_pre_s(self) -> None:
        experiment_root = self.root / "exp_bundle"
        fit_root = experiment_root / "fits" / "rank_0"
        fit_root.mkdir(parents=True, exist_ok=True)
        save_model_artifacts(
            experiment_root,
            ModelArtifacts(
                gamma_matrix=np.zeros((2, 2), dtype=float),
                t_steps=2,
                latent_rank=0,
                field_matrix=np.zeros((2, 2), dtype=float),
            ),
        )
        save_estimated_parameter_bundle(
            fit_root / "estimated_parameter_bundle.npz",
            beta=0.5,
            xi=0.1,
            eta=-0.2,
            latent_rank=0,
            t_steps=2,
            field_matrix=np.zeros((2, 2), dtype=float),
        )
        OmegaConf.save(
            OmegaConf.create(
                {
                    "estimation_params": {
                        "fixed_scalar_params": {},
                        "beta_mask_pre_s": True,
                    }
                }
            ),
            fit_root / "fit_realized_config.yaml",
        )

        bundle = load_fit_parameter_bundle(fit_root, experiment_root)
        self.assertTrue(bundle.beta_mask_pre_s)

    def test_simulate_outcomes_for_bundle_masks_beta_pre_s(self) -> None:
        field_matrix = np.zeros((2, 8), dtype=float)
        gamma_matrix = np.zeros((8, 8), dtype=float)
        z = np.vstack([np.ones(8, dtype=float), -np.ones(8, dtype=float)])
        x_0 = np.ones(8, dtype=float)
        masked_bundle = OutcomeParameterBundle(
            source_type="fit",
            source_name="rank_0",
            beta=12.0,
            xi=0.0,
            eta=0.0,
            beta_mask_pre_s=True,
            latent_rank=0,
            t_steps=2,
            field_matrix=field_matrix,
            gamma_matrix=gamma_matrix,
        )

        masked = simulate_outcomes_for_bundle(
            masked_bundle,
            x_0=x_0,
            z=z,
            gibbs_sweeps=1,
            seed=0,
            s=1,
        )
        expected_masked = simulate_outcomes_given_fixed_interventions(
            x_0=x_0,
            z=z,
            field_matrix=field_matrix,
            interaction_matrix=gamma_matrix,
            beta=12.0,
            eta=0.0,
            rng=np.random.default_rng(0),
            gibbs_sweeps=1,
            s=1,
            beta_mask_pre_s=True,
        )
        unmasked = simulate_outcomes_given_fixed_interventions(
            x_0=x_0,
            z=z,
            field_matrix=field_matrix,
            interaction_matrix=gamma_matrix,
            beta=12.0,
            eta=0.0,
            rng=np.random.default_rng(0),
            gibbs_sweeps=1,
            s=1,
            beta_mask_pre_s=False,
        )

        self.assertTrue(np.array_equal(masked, expected_masked))
        self.assertFalse(np.array_equal(masked, unmasked))

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
                    "num_samples": 8,
                    "gibbs_sweeps": 2,
                    "seed": 0,
                    "mean_abs_zscore": 0.25,
                    "max_abs_zscore": 0.50,
                    "coverage_rate": 1.0,
                    "num_statistics": 10,
                    "output_path": str(
                        (
                            self.root
                            / "exp_a"
                            / "posterior_predictive"
                            / "truth"
                            / "run_one"
                        ).resolve()
                    ),
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
                    "num_samples": 8,
                    "gibbs_sweeps": 2,
                    "seed": 0,
                    "mean_abs_zscore": 0.40,
                    "max_abs_zscore": 0.45,
                    "coverage_rate": 0.9,
                    "num_statistics": 10,
                    "output_path": str(
                        (
                            self.root
                            / "exp_a"
                            / "posterior_predictive"
                            / "fit_rank_0"
                            / "run_one"
                        ).resolve()
                    ),
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
                    "num_samples": 8,
                    "gibbs_sweeps": 2,
                    "seed": 0,
                    "mean_abs_zscore": 0.30,
                    "max_abs_zscore": 0.60,
                    "coverage_rate": 0.9,
                    "num_statistics": 10,
                    "output_path": str(
                        (
                            self.root
                            / "exp_tie"
                            / "posterior_predictive"
                            / "fit_variant_worse_max"
                            / "run_one"
                        ).resolve()
                    ),
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
                    "num_samples": 8,
                    "gibbs_sweeps": 2,
                    "seed": 0,
                    "mean_abs_zscore": 0.30,
                    "max_abs_zscore": 0.40,
                    "coverage_rate": 0.9,
                    "num_statistics": 10,
                    "output_path": str(
                        (
                            self.root
                            / "exp_tie"
                            / "posterior_predictive"
                            / "fit_variant_better_max"
                            / "run_one"
                        ).resolve()
                    ),
                },
            ]
        )
        rows = collect_predictive_rows(manifest_path)
        grouped, _ = group_and_rank_predictive_rows(rows)
        ranked = grouped[str((self.root / "exp_tie").resolve())]
        self.assertEqual(ranked[0]["source_name"], "variant_better_max")
        self.assertEqual(ranked[0]["rank_in_experiment"], 1)

    def test_write_intervention_summaries_adds_truth_metrics_and_ranking(self) -> None:
        experiment_root = self.root / "exp_counterfactual"
        truth_root = (
            experiment_root / "counterfactual" / "truth" / "all_zeros" / "default"
        )
        better_root = (
            experiment_root
            / "counterfactual"
            / "fit_better"
            / "all_zeros"
            / "default"
        )
        worse_root = (
            experiment_root
            / "counterfactual"
            / "fit_worse"
            / "all_zeros"
            / "default"
        )
        self._write_counterfactual_summary_outputs(
            truth_root,
            overall_mean=0.20,
            overall_q025=0.15,
            overall_q500=0.20,
            overall_q975=0.25,
            post_mean=0.55,
            post_q025=0.45,
            post_q500=0.55,
            post_q975=0.65,
            unit_means=[0.10, 0.60, -0.20],
            unit_q025=[-0.05, 0.40, -0.35],
            unit_q975=[0.25, 0.80, -0.05],
        )
        self._write_counterfactual_summary_outputs(
            better_root,
            overall_mean=0.23,
            overall_q025=0.18,
            overall_q500=0.23,
            overall_q975=0.28,
            post_mean=0.52,
            post_q025=0.42,
            post_q500=0.52,
            post_q975=0.62,
            unit_means=[0.12, 0.58, -0.18],
            unit_q025=[-0.02, 0.38, -0.30],
            unit_q975=[0.26, 0.78, -0.06],
        )
        self._write_counterfactual_summary_outputs(
            worse_root,
            overall_mean=0.38,
            overall_q025=0.32,
            overall_q500=0.38,
            overall_q975=0.44,
            post_mean=0.82,
            post_q025=0.72,
            post_q500=0.82,
            post_q975=0.92,
            unit_means=[0.75, -0.10, 0.35],
            unit_q025=[0.55, -0.30, 0.15],
            unit_q975=[0.95, 0.10, 0.55],
        )

        manifest_rows = [
            {
                "source_type": "truth",
                "source_name": "truth",
                "source_slug": "truth",
                "run_name": "default",
                "run_slug": "default",
                "num_samples": 4,
                "gibbs_sweeps": 1,
                "s": 2,
                "target_intervention_source": "saved_intervention",
                "target_intervention_slug": "all_zeros",
                "output_path": str(truth_root.resolve()),
            },
            {
                "source_type": "fit",
                "source_name": "better_fit",
                "source_slug": "fit_better",
                "run_name": "default",
                "run_slug": "default",
                "num_samples": 4,
                "gibbs_sweeps": 1,
                "s": 2,
                "target_intervention_source": "saved_intervention",
                "target_intervention_slug": "all_zeros",
                "output_path": str(better_root.resolve()),
            },
            {
                "source_type": "fit",
                "source_name": "worse_fit",
                "source_slug": "fit_worse",
                "run_name": "default",
                "run_slug": "default",
                "num_samples": 4,
                "gibbs_sweeps": 1,
                "s": 2,
                "target_intervention_source": "saved_intervention",
                "target_intervention_slug": "all_zeros",
                "output_path": str(worse_root.resolve()),
            },
        ]

        write_intervention_summaries(experiment_root, manifest_rows)

        summary_path = experiment_root / "intervention_summaries" / "all_zeros.csv"
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        by_name = {row["source_name"]: row for row in rows}

        self.assertIn("truth_unit_mean_squared_error_mean", rows[0])
        self.assertNotIn("truth_time_mean_abs_error", rows[0])
        self.assertEqual(float(by_name["truth"]["truth_unit_mean_squared_error_mean"]), 0.0)
        self.assertEqual(float(by_name["truth"]["truth_overall_mean_magnetization_abs_error"]), 0.0)
        self.assertEqual(by_name["truth"]["truth_rank_in_run"], "")
        self.assertEqual(by_name["truth"]["truth_is_best"], "")

        better_mse = float(by_name["better_fit"]["truth_unit_mean_squared_error_mean"])
        worse_mse = float(by_name["worse_fit"]["truth_unit_mean_squared_error_mean"])
        self.assertLess(better_mse, worse_mse)
        self.assertEqual(by_name["better_fit"]["truth_rank_in_run"], "1")
        self.assertEqual(by_name["better_fit"]["truth_is_best"], "True")
        self.assertEqual(by_name["worse_fit"]["truth_rank_in_run"], "2")
        self.assertEqual(by_name["worse_fit"]["truth_is_best"], "False")
        self.assertGreater(
            float(by_name["better_fit"]["truth_unit_mean_95_interval_coverage_rate"]),
            float(by_name["worse_fit"]["truth_unit_mean_95_interval_coverage_rate"]),
        )

    def test_write_intervention_summaries_leaves_truth_metrics_blank_without_truth_row(
        self,
    ) -> None:
        experiment_root = self.root / "exp_missing_truth"
        fit_root = (
            experiment_root / "counterfactual" / "fit_rank_0" / "all_zeros" / "default"
        )
        self._write_counterfactual_summary_outputs(
            fit_root,
            overall_mean=0.10,
            overall_q025=0.05,
            overall_q500=0.10,
            overall_q975=0.15,
            post_mean=0.20,
            post_q025=0.10,
            post_q500=0.20,
            post_q975=0.30,
            unit_means=[0.10, 0.20],
            unit_q025=[0.0, 0.10],
            unit_q975=[0.20, 0.30],
        )

        write_intervention_summaries(
            experiment_root,
            [
                {
                    "source_type": "fit",
                    "source_name": "rank_0",
                    "source_slug": "fit_rank_0",
                    "run_name": "default",
                    "run_slug": "default",
                    "num_samples": 4,
                    "gibbs_sweeps": 1,
                    "s": 1,
                    "target_intervention_source": "saved_intervention",
                    "target_intervention_slug": "all_zeros",
                    "output_path": str(fit_root.resolve()),
                }
            ],
        )

        summary_path = experiment_root / "intervention_summaries" / "all_zeros.csv"
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["truth_unit_mean_squared_error_mean"], "")
        self.assertEqual(row["truth_rank_in_run"], "")
        self.assertEqual(row["truth_is_best"], "")

    def test_write_intervention_summaries_keeps_observed_experiment_truth_fields_blank(
        self,
    ) -> None:
        experiment_root = self.root / "exp_observed"
        observed_root = (
            experiment_root
            / "posterior_predictive"
            / "fit_rank_0"
            / "default"
        )
        self._write_predictive_stats_output(
            observed_root,
            overall_mean=0.15,
            post_mean=0.25,
        )

        write_intervention_summaries(
            experiment_root,
            [
                {
                    "source_type": "fit",
                    "source_name": "rank_0",
                    "source_slug": "fit_rank_0",
                    "run_name": "default",
                    "run_slug": "default",
                    "num_samples": 4,
                    "gibbs_sweeps": 1,
                    "s": 1,
                    "target_intervention_source": "observed_experiment",
                    "target_intervention_slug": "observed_experiment",
                    "output_path": str(observed_root.resolve()),
                }
            ],
        )

        summary_path = (
            experiment_root / "intervention_summaries" / "observed_experiment.csv"
        )
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["overall_mean_magnetization_mean"], "0.15")
        self.assertEqual(row["truth_unit_mean_squared_error_mean"], "")
        self.assertEqual(row["truth_rank_in_run"], "")

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
                {
                    "experiment_name": "exp_a",
                    "source_type": "truth",
                    "variant_name": "",
                },
                {
                    "experiment_name": "exp_a",
                    "source_type": "fit",
                    "variant_name": "rank_0",
                },
            ]
        )
        generation_lookup = index_generation_rows(generation_manifest_path)
        fit_lookup = resolve_fit_lookup(fit_manifest_path)
        resolved = resolve_target_pairs(
            target_pairs_path,
            generation_lookup,
            fit_lookup,
        )
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0]["source_slug"], "truth")
        self.assertEqual(resolved[1]["source_slug"], "fit_rank_0")
        self.assertEqual(resolved[0]["intervention_source"], "observed_experiment")
        self.assertEqual(resolved[0]["intervention_slug"], "observed_experiment")

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
            resolve_target_pairs(target_pairs_path, {}, {})

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
            resolve_target_pairs(
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

    def test_run_intervention_library_writes_saved_artifacts(self) -> None:
        generation_spec_path = self.root / "generation_spec.yaml"
        intervention_spec_path = self.root / "intervention_library_spec.yaml"
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
        intervention_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    "  experiment_name: smoke_rank_0",
                    "interventions:",
                    "  - name: observed_copy",
                    "    source_kind: observed_experiment",
                    "  - name: full_on_from_s",
                    "    source_kind: full_on",
                    "    activation_scope: from_s",
                    "  - name: single_unit_2_from_step_2",
                    "    source_kind: single_unit_on",
                    "    unit_index: 2",
                    "    activation_scope: from_step",
                    "    start_step: 2",
                ]
            ),
            encoding="utf-8",
        )

        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        library_manifest = run_intervention_library(
            generation_manifest,
            intervention_spec_path,
            overwrite=True,
        )

        experiment_root = self.root / "generated" / "smoke_rank_0"
        observed_copy = load_saved_intervention_context(
            experiment_root, "observed_copy"
        )
        full_on = load_saved_intervention_context(experiment_root, "full_on_from_s")
        single_unit = load_saved_intervention_context(
            experiment_root, "single_unit_2_from_step_2"
        )

        self.assertEqual(
            Path(library_manifest),
            self.root / "generated" / "intervention_library_manifest.csv",
        )
        self.assertTrue(np.array_equal(observed_copy.z_0, np.zeros(6, dtype=float)))
        self.assertTrue(np.array_equal(full_on.z[:1, :], -np.ones((1, 6), dtype=float)))
        self.assertTrue(np.array_equal(full_on.z[1:, :], np.ones((3, 6), dtype=float)))
        self.assertEqual(full_on.s, 1)
        self.assertTrue(np.array_equal(single_unit.z[:2, 2], -np.ones(2, dtype=float)))
        self.assertTrue(np.array_equal(single_unit.z[2:, 2], np.ones(2, dtype=float)))
        self.assertTrue(
            np.array_equal(single_unit.z[:, :2], -np.ones((4, 2), dtype=float))
        )

    def test_run_posterior_predictive_writes_counterfactual_outputs(self) -> None:
        generation_spec_path = self.root / "generation_spec.yaml"
        fits_spec_path = self.root / "fits_spec.yaml"
        predictive_spec_path = self.root / "posterior_predictive_spec.yaml"
        target_pairs_path = self.root / "target_pairs.csv"
        intervention_spec_path = self.root / "intervention_library_spec.yaml"
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
                ]
            ),
            encoding="utf-8",
        )
        intervention_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    "  experiment_name: smoke_rank_0",
                    "interventions:",
                    "  - name: full_on_from_s",
                    "    source_kind: full_on",
                    "    activation_scope: from_s",
                ]
            ),
            encoding="utf-8",
        )
        target_pairs_path.write_text(
            "\n".join(
                [
                    "experiment_name,source_type,variant_name,intervention_source,intervention_name",
                    "smoke_rank_0,truth,,observed_experiment,",
                    "smoke_rank_0,fit,rank_0,saved_intervention,full_on_from_s",
                ]
            ),
            encoding="utf-8",
        )

        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        fit_manifest = run_fits(generation_manifest, fits_spec_path, overwrite=True)
        run_intervention_library(
            generation_manifest,
            intervention_spec_path,
            overwrite=True,
        )

        truth_row = run_posterior_predictive(
            generation_manifest,
            fit_manifest,
            target_pairs_path,
            predictive_spec_path,
            experiment_name="smoke_rank_0",
            source_type="truth",
            variant_name="",
            intervention_source="observed_experiment",
            intervention_name="",
            run_name="default",
            overwrite=True,
        )
        fit_row = run_posterior_predictive(
            generation_manifest,
            fit_manifest,
            target_pairs_path,
            predictive_spec_path,
            experiment_name="smoke_rank_0",
            source_type="fit",
            variant_name="rank_0",
            intervention_source="saved_intervention",
            intervention_name="full_on_from_s",
            run_name="default",
            overwrite=True,
        )
        report_outputs = refresh_and_write_posterior_predictive_reports(generation_manifest)

        experiment_root = self.root / "generated" / "smoke_rank_0"
        observed_output = (
            experiment_root
            / "posterior_predictive"
            / "truth"
            / "default"
            / "posterior_predictive_stats.csv"
        )
        counterfactual_root = (
            experiment_root
            / "counterfactual"
            / "fit_rank_0"
            / "full_on_from_s"
            / "default"
        )
        predictive_manifest = self.root / "generated" / "posterior_predictive_manifest.csv"

        self.assertEqual(
            Path(str(truth_row["output_path"])).resolve(),
            (experiment_root / "posterior_predictive" / "truth" / "default").resolve(),
        )
        self.assertEqual(
            Path(str(fit_row["output_path"])).resolve(),
            counterfactual_root.resolve(),
        )
        self.assertTrue(observed_output.exists())
        self.assertTrue(predictive_manifest.exists())
        self.assertTrue((counterfactual_root / "counterfactual_metadata.yaml").exists())
        self.assertTrue(
            (counterfactual_root / "counterfactual_sample_summaries.npz").exists()
        )
        self.assertTrue((counterfactual_root / "counterfactual_summary.csv").exists())
        self.assertTrue(
            (counterfactual_root / "counterfactual_unit_summary.csv").exists()
        )
        self.assertFalse(
            (counterfactual_root / "posterior_predictive_stats.csv").exists()
        )

        self.assertEqual(
            Path(report_outputs["manifest_path"]).resolve(),
            predictive_manifest.resolve(),
        )
        with predictive_manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            sum(row["target_intervention_source"] == "saved_intervention" for row in rows),
            1,
        )
        self.assertEqual(
            sum(row["target_intervention_source"] == "observed_experiment" for row in rows),
            1,
        )

    def test_run_posterior_predictive_reports_refresh_unified_manifest(self) -> None:
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
                    "experiment_name,source_type,variant_name,intervention_source,intervention_name",
                    "smoke_rank_0,truth,,observed_experiment,",
                    "smoke_rank_0,fit,rank_0,observed_experiment,",
                ]
            ),
            encoding="utf-8",
        )

        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        fit_manifest = run_fits(generation_manifest, fits_spec_path, overwrite=True)

        experiment_root = self.root / "generated" / "smoke_rank_0"
        truth_bundle = load_truth_parameter_bundle(experiment_root)
        fit_bundle = load_fit_parameter_bundle(
            experiment_root / "fits" / "rank_0", experiment_root
        )
        self.assertEqual(truth_bundle.field_matrix.shape, (4, 6))
        self.assertEqual(fit_bundle.field_matrix.shape, (4, 6))

        for source_type, variant_name in [("truth", ""), ("fit", "rank_0")]:
            for run_name in ["default", "longer"]:
                run_posterior_predictive(
                    generation_manifest,
                    fit_manifest,
                    target_pairs_path,
                    predictive_spec_path,
                    experiment_name="smoke_rank_0",
                    source_type=source_type,
                    variant_name=variant_name,
                    intervention_source="observed_experiment",
                    intervention_name="",
                    run_name=run_name,
                    overwrite=True,
                )
        report_outputs = refresh_and_write_posterior_predictive_reports(generation_manifest)

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
        winners_csv = (
            self.root / "generated" / "best_posterior_predictive_by_experiment.csv"
        )

        self.assertTrue(truth_default_csv.exists())
        self.assertTrue(truth_longer_csv.exists())
        self.assertTrue(fit_default_csv.exists())
        self.assertTrue(fit_longer_csv.exists())
        self.assertTrue(summary_csv.exists())
        self.assertTrue(winners_csv.exists())

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
            Path(report_outputs["manifest_path"]),
            self.root / "generated" / "posterior_predictive_manifest.csv",
        )

    @unittest.skipIf(shutil.which("bash") is None, "bash is required for shell submission test")
    def test_submit_posterior_predictive_jobs_submits_workers_and_report(self) -> None:
        bash_path = shutil.which("bash")
        if bash_path is None or "system32" in bash_path.lower():
            self.skipTest("portable bash is not available in this environment")
        target_pairs_path = self.root / "target_pairs.csv"
        predictive_spec_path = self.root / "posterior_predictive_spec.yaml"
        fake_sbatch_path = self.root / "fake_sbatch.sh"
        fake_counter_path = self.root / "fake_sbatch_counter.txt"
        fake_log_path = self.root / "fake_sbatch_log.txt"

        target_pairs_path.write_text(
            "\n".join(
                [
                    "experiment_name,source_type,variant_name,intervention_source,intervention_name",
                    "exp_a,truth,,observed_experiment,",
                ]
            ),
            encoding="utf-8",
        )
        predictive_spec_path.write_text(
            "\n".join(
                [
                    "base:",
                    "  num_samples: 2",
                    "  gibbs_sweeps: 1",
                    "  seed: 0",
                    "runs:",
                    "  - name: default",
                    "  - name: longer",
                ]
            ),
            encoding="utf-8",
        )
        fake_sbatch_path.write_text(
            "\n".join(
                [
                    "#!/bin/bash",
                    "set -euo pipefail",
                    'count="0"',
                    'if [[ -f "${FAKE_SBATCH_COUNTER}" ]]; then',
                    '  count="$(cat "${FAKE_SBATCH_COUNTER}")"',
                    "fi",
                    'count="$((count + 1))"',
                    'printf "%s" "${count}" > "${FAKE_SBATCH_COUNTER}"',
                    'printf "%s\\n" "$*" >> "${FAKE_SBATCH_LOG}"',
                    'printf "%s\\n" "job${count}"',
                ]
            ),
            encoding="utf-8",
        )
        fake_sbatch_path.chmod(0o755)

        subprocess.run(
            [
                bash_path,
                "submit_posterior_predictive_jobs.sh",
            ],
            check=True,
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "TARGET_PAIRS_PATH": str(target_pairs_path),
                "POSTERIOR_PREDICTIVE_SPEC_PATH": str(predictive_spec_path),
                "SBATCH_BIN": str(fake_sbatch_path),
                "WORKER_SCRIPT": "run_posterior_predictive_job.sh",
                "FAKE_SBATCH_COUNTER": str(fake_counter_path),
                "FAKE_SBATCH_LOG": str(fake_log_path),
            },
        )

        log_lines = fake_log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log_lines), 3)
        self.assertEqual(
            log_lines[0],
            "8|<--parsable><run_posterior_predictive_job.sh><exp_a><truth><><observed_experiment><><default>",
        )
        self.assertEqual(
            log_lines[1],
            "8|<--parsable><run_posterior_predictive_job.sh><exp_a><truth><><observed_experiment><><longer>",
        )
        self.assertIn("<--wrap>", log_lines[2])


if __name__ == "__main__":
    unittest.main()
