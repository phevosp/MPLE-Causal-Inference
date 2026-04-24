"""Small regression tests for the minimal latent-only MPLE pipeline."""

from __future__ import annotations

import shutil
import os
import subprocess
import sys
import unittest
import uuid
import csv
from pathlib import Path

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
from data.USCountyVaccination.experiment_artifacts import (
    create_config as create_us_county_config,
    save_experiment as save_us_county_experiment,
)
from intervention_utils import load_saved_intervention_context
from io_utils import io_path
from loading_utils import (
    OutcomeParameterBundle,
    load_experiment_panel_context,
    load_fit_parameter_bundle,
    load_truth_parameter_bundle,
)
from mple import (
    _build_fit_eval_context,
    _evaluate_factorized_loss,
    _evaluate_full_field_loss,
    _evaluate_scalar_only_loss,
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
    load_true_parameters,
    parameter_names,
    project_latent_field,
    save_model_artifacts,
    unpack_theta,
)
from pipeline_specs import validate_fits_spec
from posterior_predictive_utils import (
    compute_panel_statistics,
    simulate_outcomes_for_bundle,
    summarize_predictive_statistics,
)
from report_posterior_predictive import (
    collect_predictive_rows,
    group_and_rank_predictive_rows,
    refresh_and_write_posterior_predictive_reports,
)
from report_parameter_recovery_detailed import (
    collect_fit_rows,
    group_and_rank_fit_rows,
    write_fit_reports,
)
from run_fit_pipeline import build_fit_config, infer_panel_dimensions, run_fits
from run_generation_pipeline import run_generation
from run_intervention_library import run_intervention_library
from posterior_predictive_job_utils import (
    index_generation_rows,
    resolve_fit_lookup,
    resolve_target_pairs,
)
from run_posterior_predictive import run_posterior_predictive
from run_uscounty_sensitivity_analysis import (
    materialize_sensitivity_experiments,
    write_sensitivity_fit_spec,
)


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


class USCountyVaccinationSharedPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "experiments" / f".tmp_uscounty_{uuid.uuid4().hex}"
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
            field_basis_mode="zero",
            field_basis_names=(),
            model_field_mode="uniform",
            latent_rank=0,
            latent_B=1.0,
            state_scope_label="Mainland US counties with total_population >= 2000",
            tau_zero_mean=False,
            tau_smoothness_lambda=0.0,
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

    def test_us_county_sensitivity_materializes_start_week_slices(self) -> None:
        _, manifest_path = self._write_us_county_experiment()
        output_root = self.root / "sensitivity"

        sensitivity_manifest = materialize_sensitivity_experiments(
            source_manifest_path=manifest_path,
            output_root=output_root,
            start_dates=["2021-01-23"],
            overwrite=True,
        )
        fit_spec_path = write_sensitivity_fit_spec(
            output_root=output_root,
            latent_ranks=[0, 2],
            b_values=[1.0, 5.0],
            steps=3,
            tol=1.0e-6,
            seed=9,
            lambda_nuclear_values=[0.01],
        )

        with sensitivity_manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["T"], "2")
        self.assertEqual(rows[0]["s"], "0")
        self.assertEqual(rows[0]["sensitivity_start_index"], "2")
        self.assertEqual(rows[0]["sensitivity_start_week_end_date"], "2021-01-23")

        derived_root = Path(rows[0]["experiment_path"])
        self.assertTrue((derived_root / "panel_data.npz").exists())
        self.assertTrue(
            np.array_equal(
                np.load(derived_root / "x_0.npy"),
                np.array([1, 1, -1, -1], dtype=np.int8),
            )
        )
        derived_dims = infer_panel_dimensions(derived_root)
        self.assertEqual(derived_dims, {"N": 4, "T": 2, "s": 0})

        fit_spec = OmegaConf.load(fit_spec_path)
        self.assertEqual(len(fit_spec.variants), 6)
        self.assertEqual(fit_spec.base.optimizer.steps, 3)
        self.assertEqual(fit_spec.base.optimizer.n_starts, 1)
        self.assertEqual(fit_spec.variants[-1].optimizer_mode, "nuclear_norm")
        self.assertAlmostEqual(float(fit_spec.variants[-1].lambda_nuclear), 0.01)

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
        summary_md = experiment_root / "posterior_predictive_summary.md"
        winners_csv = (
            self.root / "generated" / "best_posterior_predictive_by_experiment.csv"
        )
        winners_md = (
            self.root / "generated" / "best_posterior_predictive_by_experiment.md"
        )

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
        self.assertIn("run_posterior_predictive_job.sh exp_a truth", log_lines[0])
        self.assertIn("default", log_lines[0])
        self.assertIn("run_posterior_predictive_job.sh exp_a truth", log_lines[1])
        self.assertIn("longer", log_lines[1])
        self.assertIn("--wrap", log_lines[2])


if __name__ == "__main__":
    unittest.main()
