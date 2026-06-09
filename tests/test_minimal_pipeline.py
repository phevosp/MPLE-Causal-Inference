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
from types import SimpleNamespace
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
    sample_low_rank_probability_interventions,
    simulate_outcomes_given_fixed_interventions,
)
import build_splits as uscounty_splits
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
from utils.t6_intervention_utils import load_saved_intervention_context
from utils.t0_config_utils import deep_merge_mappings, load_yaml_mapping
from utils.t0_csv_utils import read_csv_rows as read_csv_manifest, write_csv_rows
from utils.t0_path_utils import io_path
from utils.t0_string_utils import slugify
from utils.t5_parameter_bundles import (
    OutcomeParameterBundle,
    load_fit_parameter_bundle,
    load_truth_parameter_bundle,
    save_estimated_parameter_bundle,
)
from utils.t5_experiment_context import (
    infer_panel_dimensions,
    load_experiment_panel_context,
)
from utils.t6_fit_materialization import (
    build_fit_config,
    execute_fit_root,
    materialize_fit_root,
)
from mple import (
    _build_fit_eval_context,
    _compute_h_x,
    _evaluate_factorized_loss,
    _evaluate_full_field_loss,
    _evaluate_scalar_only_loss,
    _project_node_factor_columns_to_l2_ball,
    evaluate_mple_loss_from_parts,
    fit_mple,
    pseudo_nll,
)
from utils.t3_model_artifacts import (
    ModelArtifacts,
    build_fit_model_artifacts,
    load_model_artifacts,
    save_model_artifacts,
)
from utils.t3_field_generation import (
    SpectralLowRankStructure,
    build_synthetic_field,
    leading_svd_low_rank_structure,
    sample_spectral_low_rank_structure,
    SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK,
)
from utils.t3_interaction_matrices import (
    compose_interaction_matrix,
    interaction_effect,
    interaction_term,
    interaction_matrix_infinity_norm,
)
from utils.t3_field_operations import (
    compose_latent_field_matrix,
    latent_field_bound_norm,
    project_latent_field,
)
from utils.t4_scalar_parameters import get_xi
from utils.t4_parameter_packing import (
    parameter_names,
    unpack_theta,
    load_true_parameters,
)
from utils.t2_normalization import normalize_matrix_max_abs
from utils.t6_pipeline_spec_utils import (
    expand_named_entries,
    validate_cv_spec,
    validate_fits_spec,
)
import utils.t6_split_engine as cv_folds
import run_posterior_predictive as posterior_predictive_runner
import run_cv_folds as cv_runner
import utils.t7_validation_metrics as validation_metrics
from utils.t8_posterior_predictive_sim import (
    compute_panel_statistics,
    compute_counterfactual_sample_summary,
    simulate_outcomes_for_bundle,
)
from utils.t8_posterior_predictive_reporting import (
    summarize_observed_mean_statistics,
    summarize_predictive_statistics,
)
from utils.t8_output_writers import (
    write_counterfactual_summary_tables,
    write_observed_predictive_summary_tables,
)
from utils.t6_intervention_utils import derive_pre_intervention_steps
from utils.t8_posterior_predictive_reporting import (
    collect_predictive_rows,
    group_and_rank_predictive_rows,
    refresh_and_write_posterior_predictive_reports,
    write_intervention_summaries,
)
from utils.t8_parameter_recovery_reporting import (
    collect_fit_rows,
    group_and_rank_fit_rows,
    write_fit_reports,
)
from run_fit_pipeline import (
    refresh_fit_manifest,
    refresh_train_fit_manifest,
    run_fit_request,
    run_fits,
    run_train_fit_request,
    write_train_fit_requests,
    write_fit_requests,
)
from run_generation_pipeline import (
    refresh_generation_manifest,
    run_generation,
    run_generation_request,
    write_generation_requests,
)
from run_intervention_library import run_intervention_library
from utils.t6_posterior_predictive_manifest import (
    index_generation_rows,
    resolve_fit_lookup,
    resolve_target_pairs,
)
from utils.t6_posterior_predictive_summary import (
    build_manifest_row,
    manifest_row_from_metadata,
)
from run_posterior_predictive import run_posterior_predictive
from run_test_evaluation import run_test_evaluation
from utils.t6_split_management import (
    load_model_selection_split_masks,
    load_outer_training_split_masks,
    load_outer_test_split_masks,
)


def base_config() -> object:
    return OmegaConf.create(
        {
            "global_params": {
                "N": 4,
                "T": 3,
                "B": 1.0,
                "field_params": {},
            },
            "estimation_params": {
                "xi": 0.25,
                "beta": 0.1,
                "eta": 0.2,
            },
            "generation_params": {
                "seed": 0,
                "gibbs_sweeps": 1,
                "intervention_mode": "low_rank_probability",
                "intervention_params": {},
            },
        }
    )


class PipelineSpecUtilityTests(unittest.TestCase):
    def _workspace_temp_root(self) -> Path:
        root = REPO_ROOT / "experiments" / f".tmp_pipeline_utils_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        return root

    def test_slugify_uses_fallback_for_empty_text(self) -> None:
        self.assertEqual(slugify("  "), "item")
        self.assertEqual(slugify("###", fallback="experiment"), "experiment")

    def test_deep_merge_mappings_merges_nested_dicts_without_mutation(self) -> None:
        base = {"optimizer": {"steps": 10, "tol": 1.0e-6}, "name": "base"}
        override = {"optimizer": {"steps": 25}, "latent_rank": 2}

        merged = deep_merge_mappings(base, override)

        self.assertEqual(
            merged,
            {
                "optimizer": {"steps": 25, "tol": 1.0e-6},
                "name": "base",
                "latent_rank": 2,
            },
        )
        self.assertEqual(base["optimizer"]["steps"], 10)

    def test_load_yaml_mapping_returns_plain_dict(self) -> None:
        temp_root = self._workspace_temp_root()
        try:
            spec_path = temp_root / "spec.yaml"
            spec_path.write_text("base:\n  optimizer:\n    steps: 5\n", encoding="utf-8")

            payload = load_yaml_mapping(spec_path)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(payload, {"base": {"optimizer": {"steps": 5}}})

    def test_write_and_read_csv_rows_preserve_first_seen_column_order(self) -> None:
        temp_root = self._workspace_temp_root()
        try:
            csv_path = temp_root / "manifest.csv"
            write_csv_rows(
                csv_path,
                [
                    {"b": "2", "a": "1"},
                    {"c": "3", "a": "4"},
                ],
            )

            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            rows = read_csv_manifest(csv_path)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(header, "b,a,c")
        self.assertEqual(rows[1]["c"], "3")

    def test_expand_named_entries_deep_merges_base_and_adds_slug(self) -> None:
        temp_root = self._workspace_temp_root()
        try:
            spec_path = temp_root / "generation_spec.yaml"
            spec_path.write_text(
                "\n".join(
                    [
                        "base:",
                        "  optimizer:",
                        "    steps: 10",
                        "    tol: 1.0e-6",
                        "entries:",
                        "  - name: Rank 2",
                        "    optimizer:",
                        "      steps: 20",
                    ]
                ),
                encoding="utf-8",
            )

            entries = expand_named_entries(spec_path, "entries")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(entries[0]["optimizer"]["steps"], 20)
        self.assertEqual(entries[0]["optimizer"]["tol"], 1.0e-6)
        self.assertEqual(entries[0]["slug"], "rank_2")


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

    def test_interaction_term_uses_same_canonical_operator_as_sampling(self) -> None:
        x = np.array([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], dtype=float)
        gamma = np.array(
            [
                [3.0, 2.0, 0.0],
                [0.0, -5.0, 4.0],
                [1.0, 0.0, 7.0],
            ],
            dtype=float,
        )
        xi = 0.25
        interaction_matrix = compose_interaction_matrix(xi, gamma)
        expected = np.asarray(x @ interaction_matrix.T, dtype=float)

        self.assertTrue(np.allclose(interaction_term(x, xi, gamma), expected))
        self.assertTrue(
            np.allclose(interaction_effect(x, gamma), interaction_term(x, 1.0, gamma))
        )

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
            fixed_z,
            z_0,
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
            beta_mask_post_e=False,
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
            fixed_z,
            None,
        )
        predictive_x = simulate_outcomes_for_bundle(
            bundle,
            x_0=x_0,
            z=fixed_z,
            gibbs_sweeps=int(config.generation_params.gibbs_sweeps),
            seed=seed,
        )

        self.assertTrue(np.array_equal(generated_x, predictive_x))

    def test_posterior_predictive_sample_seed_depends_on_target_identity(self) -> None:
        run_spec = {"name": "default", "slug": "default", "seed": 7}
        target_a = {
            "experiment_row": {"experiment_slug": "exp_a"},
            "source_type": "truth",
            "source_slug": "truth",
            "intervention_source": "observed_experiment",
            "intervention_slug": "observed_experiment",
        }
        target_b = {
            "experiment_row": {"experiment_slug": "exp_b"},
            "source_type": "truth",
            "source_slug": "truth",
            "intervention_source": "observed_experiment",
            "intervention_slug": "observed_experiment",
        }

        seed_a = posterior_predictive_runner._posterior_sample_seed(
            target=target_a,
            run_spec=run_spec,
            sample_index=0,
        )
        seed_b = posterior_predictive_runner._posterior_sample_seed(
            target=target_b,
            run_spec=run_spec,
            sample_index=0,
        )
        seed_a_repeat = posterior_predictive_runner._posterior_sample_seed(
            target=target_a,
            run_spec=run_spec,
            sample_index=0,
        )

        self.assertNotEqual(seed_a, seed_b)
        self.assertEqual(seed_a, seed_a_repeat)

    def test_sample_spectral_low_rank_structure_returns_orthonormal_factors(self) -> None:
        structure = sample_spectral_low_rank_structure(
            n_nodes=5,
            t_steps=4,
            singular_values=np.array([1.0, 0.7], dtype=float),
            rng=np.random.default_rng(123),
        )

        self.assertEqual(structure.matrix.shape, (4, 5))
        self.assertEqual(structure.node_factors.shape, (5, 2))
        self.assertEqual(structure.time_factors.shape, (4, 2))
        self.assertTrue(
            np.allclose(
                structure.node_factors.T @ structure.node_factors,
                np.eye(2),
                atol=1e-10,
            )
        )
        self.assertTrue(
            np.allclose(
                structure.time_factors.T @ structure.time_factors,
                np.eye(2),
                atol=1e-10,
            )
        )
        self.assertLessEqual(np.linalg.matrix_rank(structure.matrix), 2)

    def test_sample_spectral_low_rank_structure_zero_singular_values_zeroes_matrix(self) -> None:
        structure = sample_spectral_low_rank_structure(
            n_nodes=4,
            t_steps=3,
            singular_values=np.zeros(2, dtype=float),
            rng=np.random.default_rng(0),
        )
        normalized = normalize_matrix_max_abs(structure.matrix, max_abs=1.0)

        self.assertTrue(np.allclose(structure.matrix, 0.0))
        self.assertTrue(np.allclose(normalized, 0.0))

    def test_positive_rank_latent_field_is_realized(self) -> None:
        config = base_config()
        config.global_params.field_params = {"singular_values": [1.0, 0.7]}
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
        self.assertGreater(float(np.sqrt(np.mean(field_matrix**2))), 0.0)

    def test_generated_latent_field_uses_target_rms_scaling(self) -> None:
        config = base_config()
        config.global_params.B = 0.5
        config.global_params.field_params = {"singular_values": [1.0, 0.7]}
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
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(field_matrix**2))),
            target_rms,
            places=12,
        )

    def test_generated_latent_field_honors_custom_target_rms_fraction(self) -> None:
        config = base_config()
        config.global_params.B = 2.0
        config.global_params.field_params = {
            "singular_values": [1.0, 0.7],
            "target_rms_fraction": 0.15,
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
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(field_matrix**2))),
            0.3,
            places=12,
        )

    def test_low_rank_probability_interventions_produce_valid_probabilities(self) -> None:
        config = base_config()
        config.global_params.N = 4
        config.global_params.T = 4
        config.generation_params.intervention_mode = "low_rank_probability"
        config.generation_params.intervention_params = {
            "singular_values": [1.0, 0.7],
            "probability_amplitude": 0.3,
        }
        artifacts = sample_low_rank_probability_interventions(config)

        self.assertTrue(np.all(np.isin(artifacts.z, (-1.0, 1.0))))
        self.assertTrue(np.array_equal(artifacts.z_0, np.zeros(4, dtype=float)))
        self.assertTrue(np.all(artifacts.probability_matrix >= 0.0))
        self.assertTrue(np.all(artifacts.probability_matrix <= 1.0))
        self.assertEqual(artifacts.low_rank_structure.matrix.shape, (4, 4))
        self.assertEqual(artifacts.low_rank_structure.node_factors.shape, (4, 2))
        self.assertEqual(artifacts.low_rank_structure.time_factors.shape, (4, 2))

    def test_generate_data_low_rank_probability_matches_fixed_intervention_pipeline(self) -> None:
        config = base_config()
        config.global_params.N = 4
        config.global_params.T = 4
        config.generation_params.intervention_mode = "low_rank_probability"
        config.generation_params.gibbs_sweeps = 2
        config.generation_params.intervention_params = {
            "singular_values": [1.0, 0.7],
            "probability_amplitude": 0.25,
        }
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
        intervention_artifacts = sample_low_rank_probability_interventions(config)
        seed = 222

        generated_x, generated_z, returned_z_0 = generate_data(
            config,
            artifacts,
            x_0,
            np.random.default_rng(seed),
            intervention_artifacts.z,
            intervention_artifacts.z_0,
        )
        expected_x = simulate_outcomes_given_fixed_interventions(
            x_0=x_0,
            z=intervention_artifacts.z,
            field_matrix=artifacts.field_matrix,
            interaction_matrix=compose_interaction_matrix(
                get_xi(config), artifacts.gamma_matrix
            ),
            beta=float(config.estimation_params.beta),
            eta=float(config.estimation_params.eta),
            rng=np.random.default_rng(seed),
            gibbs_sweeps=int(config.generation_params.gibbs_sweeps),
        )

        self.assertTrue(np.array_equal(generated_z, intervention_artifacts.z))
        self.assertTrue(np.array_equal(returned_z_0, intervention_artifacts.z_0))
        self.assertTrue(np.array_equal(generated_x, expected_x))

    def test_low_rank_probability_rank_zero_defaults_to_half_probabilities(self) -> None:
        config = base_config()
        config.generation_params.intervention_mode = "low_rank_probability"
        config.generation_params.intervention_params = {"singular_values": [], "probability_amplitude": 0.5}
        artifacts = sample_low_rank_probability_interventions(config)

        self.assertTrue(np.allclose(artifacts.low_rank_structure.matrix, 0.0))
        self.assertTrue(np.allclose(artifacts.probability_matrix, 0.5))

    def test_derive_pre_intervention_steps_uses_first_treated_row(self) -> None:
        z = np.array(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
            ],
            dtype=float,
        )
        untreated = -np.ones((4, 3), dtype=float)

        self.assertEqual(derive_pre_intervention_steps(z), 1)
        self.assertEqual(derive_pre_intervention_steps(untreated), 4)

    def test_confounded_low_rank_field_reuses_intervention_factors(self) -> None:
        config = base_config()
        config.global_params.B = 1.0
        config.global_params.N = 4
        config.global_params.T = 4
        config.global_params.field_mode = SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK
        config.global_params.field_params = {"singular_values": [1.0, 0.5]}
        config.generation_params.intervention_params = {
            "singular_values": [1.0, 0.7],
            "probability_amplitude": 0.3,
        }
        intervention_artifacts = sample_low_rank_probability_interventions(config)
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        field_artifacts = build_synthetic_field(
            config,
            gamma,
            intervention_structure=intervention_artifacts.low_rank_structure,
        )

        field_matrix = np.asarray(field_artifacts.field_matrix, dtype=float)
        expected_unscaled = (
            intervention_artifacts.low_rank_structure.time_factors
            * np.array([1.0, 0.5], dtype=float)[None, :]
        ) @ intervention_artifacts.low_rank_structure.node_factors.T
        expected_field = normalize_matrix_max_abs(expected_unscaled, max_abs=1.0)
        expected_field = expected_field * (
            (0.4 * float(config.global_params.B))
            / float(np.sqrt(np.mean(expected_field**2)))
        )

        self.assertEqual(field_artifacts.latent_rank, 2)
        self.assertEqual(field_matrix.shape, (4, 4))
        self.assertTrue(np.allclose(field_matrix, expected_field))

    def test_partial_confounded_low_rank_field_reuses_only_shared_intervention_factors(self) -> None:
        config = base_config()
        config.global_params.B = 1.0
        config.global_params.N = 4
        config.global_params.T = 4
        config.global_params.field_mode = SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK
        config.global_params.field_params = {
            "singular_values": [2.0, 0.4],
            "shared_rank": 1,
        }
        config.generation_params.intervention_params = {
            "singular_values": [1.0, 0.7],
            "probability_amplitude": 0.3,
        }
        intervention_artifacts = sample_low_rank_probability_interventions(config)
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        field_artifacts = build_synthetic_field(
            config,
            gamma,
            intervention_structure=intervention_artifacts.low_rank_structure,
        )

        field_matrix = np.asarray(field_artifacts.field_matrix, dtype=float)
        u, _, vt = np.linalg.svd(field_matrix, full_matrices=False)
        shared_time = intervention_artifacts.low_rank_structure.time_factors[:, 0]
        shared_node = intervention_artifacts.low_rank_structure.node_factors[:, 0]

        self.assertEqual(field_artifacts.latent_rank, 2)
        self.assertAlmostEqual(abs(float(np.dot(u[:, 0], shared_time))), 1.0, places=10)
        self.assertAlmostEqual(abs(float(np.dot(vt[0, :], shared_node))), 1.0, places=10)
        self.assertAlmostEqual(abs(float(np.dot(u[:, 1], shared_time))), 0.0, places=10)
        self.assertAlmostEqual(abs(float(np.dot(vt[1, :], shared_node))), 0.0, places=10)

    def test_partial_confounded_low_rank_field_uses_fixed_intervention_svd_for_shared_block(self) -> None:
        config = base_config()
        config.global_params.B = 1.0
        config.global_params.N = 4
        config.global_params.T = 4
        config.global_params.field_mode = SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK
        config.global_params.field_params = {
            "singular_values": [2.0, 0.4],
            "shared_rank": 1,
        }
        fixed_z = np.array(
            [
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, 1.0],
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
        fixed_u, fixed_s, fixed_vt = np.linalg.svd(fixed_z, full_matrices=False)
        intervention_structure = SpectralLowRankStructure(
            node_factors=np.asarray(fixed_vt[:2, :].T, dtype=float),
            time_factors=np.asarray(fixed_u[:, :2], dtype=float),
            singular_values=np.asarray(fixed_s[:2], dtype=float),
            matrix=np.asarray((fixed_u[:, :2] * fixed_s[:2][None, :]) @ fixed_vt[:2, :], dtype=float),
        )
        field_artifacts = build_synthetic_field(
            config,
            gamma,
            intervention_structure=intervention_structure,
        )

        field_matrix = np.asarray(field_artifacts.field_matrix, dtype=float)
        u, _, vt = np.linalg.svd(field_matrix, full_matrices=False)

        self.assertEqual(field_artifacts.latent_rank, 2)
        self.assertAlmostEqual(abs(float(np.dot(u[:, 0], fixed_u[:, 0]))), 1.0, places=10)
        self.assertAlmostEqual(abs(float(np.dot(vt[0, :], fixed_vt[0, :]))), 1.0, places=10)
        self.assertAlmostEqual(abs(float(np.dot(u[:, 1], fixed_u[:, 0]))), 0.0, places=10)
        self.assertAlmostEqual(abs(float(np.dot(vt[1, :], fixed_vt[0, :]))), 0.0, places=10)

    def test_partial_confounding_rejects_invalid_shared_rank(self) -> None:
        config = base_config()
        config.global_params.N = 4
        config.global_params.T = 4
        config.global_params.field_mode = SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK
        config.generation_params.intervention_params = {
            "singular_values": [1.0, 0.7],
            "probability_amplitude": 0.3,
        }
        intervention_artifacts = sample_low_rank_probability_interventions(config)
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        config.global_params.field_params = {
            "singular_values": [2.0, 0.4],
            "shared_rank": 3,
        }
        with self.assertRaisesRegex(ValueError, "must not exceed the total field rank"):
            build_synthetic_field(
                config,
                gamma,
                intervention_structure=intervention_artifacts.low_rank_structure,
            )

        config.global_params.field_params = {
            "singular_values": [2.0, 0.4, 0.2],
            "shared_rank": 3,
        }
        with self.assertRaisesRegex(
            ValueError,
            "must not exceed the available intervention basis rank",
        ):
            build_synthetic_field(
                config,
                gamma,
                intervention_structure=intervention_artifacts.low_rank_structure,
            )

    def test_shared_rank_requires_confounded_field_mode(self) -> None:
        config = base_config()
        config.global_params.N = 4
        config.global_params.T = 4
        config.global_params.field_mode = "random_low_rank"
        config.global_params.field_params = {
            "singular_values": [1.0, 0.5],
            "shared_rank": 1,
        }
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        with self.assertRaisesRegex(ValueError, "shared_rank is only valid"):
            build_synthetic_field(config, gamma)

    def test_confounded_low_rank_rejects_rank_deficient_fixed_intervention_basis(self) -> None:
        config = base_config()
        config.global_params.N = 4
        config.global_params.T = 4
        config.global_params.field_mode = SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK
        config.global_params.field_params = {
            "singular_values": [2.0, 0.4],
            "shared_rank": 2,
        }
        fixed_z = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
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
        intervention_structure = leading_svd_low_rank_structure(fixed_z, 2)

        with self.assertRaisesRegex(
            ValueError,
            "must not exceed the available intervention basis rank",
        ):
            build_synthetic_field(
                config,
                gamma,
                intervention_structure=intervention_structure,
            )

    def test_generation_pipeline_confounded_low_rank_reuses_fixed_intervention_svd(self) -> None:
        root = REPO_ROOT / "experiments" / f".tmp_fixed_z_confounding_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            gamma = np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
            fixed_z = np.array(
                [
                    [1.0, 1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0, -1.0],
                    [-1.0, 1.0, -1.0, 1.0],
                    [-1.0, -1.0, 1.0, 1.0],
                ],
                dtype=float,
            )
            z_0 = -np.ones(4, dtype=float)
            gamma_path = root / "gamma.npy"
            panel_path = root / "panel_data.npz"
            z0_path = root / "z_0.npy"
            spec_path = root / "generation_spec.yaml"
            np.save(gamma_path, gamma)
            np.savez(panel_path, x=np.zeros_like(fixed_z), z=fixed_z)
            np.save(z0_path, z_0)

            spec_path.write_text(
                "\n".join(
                    [
                        "base:",
                        f"  experiment_root: {root.as_posix()}/generated",
                        f"  manifest_path: {root.as_posix()}/generated/generation_manifest.csv",
                        "  dimensions:",
                        "    N: 4",
                        "    T: 4",
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
                        f"      panel_path: {panel_path.as_posix()}",
                        f"      z0_path: {z0_path.as_posix()}",
                        f"      artifact_dir: {root.as_posix()}",
                        "      shared_panel_dir: null",
                        "      outcome_code: null",
                        "      intervention_code: null",
                        "      lag_code: null",
                        "      trim_scope: test",
                        "  truth:",
                        "    B: 1.5",
                        f"    field_mode: {SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK}",
                        "    field_params:",
                        "      singular_values: [1.0, 0.5]",
                        "      target_rms_fraction: 0.2",
                        "    scalars:",
                        "      beta: 0.2",
                        "      xi: 0.1",
                        "      eta: 0.05",
                        "experiments:",
                        "  - name: fixed_z_confounding_smoke",
                    ]
                ),
                encoding="utf-8",
            )

            run_generation(spec_path, overwrite=True)
            experiment_root = root / "generated" / "fixed_z_confounding_smoke"
            with np.load(experiment_root / "field_artifacts.npz", allow_pickle=False) as data:
                field_matrix = np.asarray(data["field_matrix"], dtype=float)

            u, _, vt = np.linalg.svd(fixed_z, full_matrices=False)
            expected_unscaled = (u[:, :2] * np.array([1.0, 0.5], dtype=float)[None, :]) @ vt[:2, :]
            expected_field = normalize_matrix_max_abs(expected_unscaled, max_abs=1.0)
            expected_field = expected_field * (
                (0.2 * 1.5) / float(np.sqrt(np.mean(expected_field**2)))
            )

            self.assertTrue((experiment_root / "panel_data.npz").exists())
            self.assertTrue(np.allclose(field_matrix, expected_field))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_quickstart_generation_spec_includes_rank_two_example(self) -> None:
        experiments = expand_named_entries(
            REPO_ROOT / "data" / "configs" / "quickstart_generation_spec.yaml",
            "experiments",
        )
        quickstart_spec = next(
            experiment
            for experiment in experiments
            if experiment["name"] == "quickstart_rank_2"
        )
        self.assertEqual(
            quickstart_spec["truth"]["field_mode"],
            "random_low_rank",
        )
        self.assertEqual(
            quickstart_spec["intervention"]["generator"],
            "low_rank_probability",
        )
        self.assertEqual(quickstart_spec["truth"]["field_params"]["singular_values"], [1.0, 0.7])

    def test_generation_spec_rejects_removed_truth_latent_rank(self) -> None:
        spec_path = REPO_ROOT / "experiments" / f".tmp_removed_truth_rank_{uuid.uuid4().hex}.yaml"
        try:
            spec_path.write_text(
                "\n".join(
                    [
                        "base:",
                        f"  experiment_root: {(REPO_ROOT / 'experiments').as_posix()}/tmp_removed_truth_rank",
                        f"  manifest_path: {(REPO_ROOT / 'experiments').as_posix()}/tmp_removed_truth_rank/generation_manifest.csv",
                        "  dimensions:",
                        "    N: 4",
                        "    T: 3",
                        "  generation:",
                        "    gibbs_sweeps: 1",
                        "    seed: 0",
                        "  x0:",
                        "    generator: bernoulli",
                        "    params:",
                        "      p: 0.5",
                        "      fixed_val: null",
                        "  graph:",
                        "    source: generated",
                        "    generator: empty",
                        "    params: {}",
                        "    artifact: {}",
                        "  intervention:",
                        "    source: generated",
                        "    generator: low_rank_probability",
                        "    params:",
                        "      singular_values: []",
                        "      probability_amplitude: 0.5",
                        "    artifact: {}",
                        "  truth:",
                        "    B: 1.0",
                        "    latent_rank: 2",
                        "    field_mode: random_low_rank",
                        "    field_params:",
                        "      singular_values: [1.0, 0.7]",
                        "    scalars:",
                        "      beta: 0.0",
                        "      xi: 0.0",
                        "      eta: 0.0",
                        "experiments:",
                        "  - name: bad_truth_rank",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "truth.latent_rank has been removed"):
                run_generation(spec_path, overwrite=True)
        finally:
            spec_path.unlink(missing_ok=True)

    def test_generation_pipeline_smoke_with_confounding_field_mode(self) -> None:
        root = REPO_ROOT / "experiments" / f".tmp_gen_confounding_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            gamma_path = root / "gamma.npy"
            spec_path = root / "generation_spec.yaml"
            fits_spec_path = root / "fits_spec.yaml"
            gamma = np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
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
                        "    source: generated",
                        "    generator: low_rank_probability",
                        "    params:",
                        "      singular_values: [1.0, 0.7]",
                        "      probability_amplitude: 0.3",
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
                        f"    field_mode: {SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK}",
                        "    field_params:",
                        "      singular_values: [1.0, 0.5]",
                        "    scalars:",
                        "      beta: 0.2",
                        "      xi: 0.1",
                        "      eta: 0.05",
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
            self.assertTrue(
                (experiment_root / "intervention_generation_artifacts.npz").exists()
            )
            self.assertTrue((experiment_root / "fits" / "rank_0" / "mple_summary.csv").exists())
            self.assertEqual(
                Path(fit_manifest),
                root / "generated" / "fit_manifest.csv",
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_generation_pipeline_partial_confounding_writes_metadata(self) -> None:
        root = REPO_ROOT / "experiments" / f".tmp_gen_partial_confounding_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            gamma_path = root / "gamma.npy"
            spec_path = root / "generation_spec.yaml"
            gamma = np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
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
                        "    source: generated",
                        "    generator: low_rank_probability",
                        "    params:",
                        "      singular_values: [1.0, 0.7]",
                        "      probability_amplitude: 0.3",
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
                        f"    field_mode: {SYNTHETIC_FIELD_MODE_CONFOUNDED_LOW_RANK}",
                        "    field_params:",
                        "      singular_values: [2.0, 0.4]",
                        "      shared_rank: 1",
                        "    scalars:",
                        "      beta: 0.2",
                        "      xi: 0.1",
                        "      eta: 0.05",
                        "experiments:",
                        "  - name: partial_confounding_smoke",
                    ]
                ),
                encoding="utf-8",
            )

            run_generation(spec_path, overwrite=True)
            experiment_root = root / "generated" / "partial_confounding_smoke"
            metadata = OmegaConf.to_container(
                OmegaConf.load(experiment_root / "experiment_metadata.yaml"),
                resolve=True,
            )

            self.assertEqual(metadata["field_shared_rank"], 1)
            self.assertEqual(metadata["field_nonshared_rank"], 1)
            self.assertEqual(
                metadata["field_shared_basis_source"],
                "generated_intervention_basis",
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
        from utils.t8_parameter_recovery_reporting import latent_diagnostics

        fit_root = REPO_ROOT / "experiments" / f".tmp_latent_diag_{uuid.uuid4().hex}"
        fit_root.mkdir(parents=True, exist_ok=True)
        try:
            estimated_field = np.array([[2.0, -1.0], [2.0, -1.0]], dtype=float)
            true_field = np.array([[0.25, -0.75], [0.5, -0.5]], dtype=float)
            np.savez(fit_root / "estimated_field_artifacts.npz", field_matrix=estimated_field)
            np.savez(fit_root / "true_field_artifacts.npz", field_matrix=true_field)

            row = latent_diagnostics(fit_root)
            estimated_singular_values = np.linalg.svd(
                estimated_field, compute_uv=False
            )
            true_singular_values = np.linalg.svd(true_field, compute_uv=False)

            self.assertEqual(row["estimated_field_max_abs_entry"], 2.0)
            self.assertEqual(row["estimated_field_rank"], 1)
            self.assertAlmostEqual(
                float(row["estimated_field_frobenius_norm"]),
                float(np.linalg.norm(estimated_field, ord="fro")),
            )
            self.assertAlmostEqual(
                float(row["estimated_field_nuclear_norm"]),
                float(np.sum(estimated_singular_values)),
            )
            self.assertEqual(row["estimated_singular_value_count"], 1)
            self.assertAlmostEqual(float(row["estimated_u_frobenius_norm"]), 1.0)
            self.assertAlmostEqual(float(row["estimated_v_frobenius_norm"]), 1.0)
            self.assertAlmostEqual(
                float(row["estimated_sv_1"]), float(estimated_singular_values[0])
            )
            self.assertNotIn("estimated_sv_2", row)
            self.assertEqual(row["true_field_max_abs_entry"], 0.75)
            self.assertEqual(row["true_field_rank"], int(np.linalg.matrix_rank(true_field)))
            self.assertAlmostEqual(
                float(row["true_field_frobenius_norm"]),
                float(np.linalg.norm(true_field, ord="fro")),
            )
            self.assertAlmostEqual(
                float(row["true_field_nuclear_norm"]),
                float(np.sum(true_singular_values)),
            )
            self.assertEqual(
                row["true_singular_value_count"], int(np.linalg.matrix_rank(true_field))
            )
            self.assertAlmostEqual(
                float(row["true_u_frobenius_norm"]),
                float(np.sqrt(np.linalg.matrix_rank(true_field))),
            )
            self.assertAlmostEqual(
                float(row["true_v_frobenius_norm"]),
                float(np.sqrt(np.linalg.matrix_rank(true_field))),
            )
            self.assertAlmostEqual(float(row["true_sv_1"]), float(true_singular_values[0]))
            self.assertAlmostEqual(float(row["true_sv_2"]), float(true_singular_values[1]))
            self.assertNotIn("true_sv_3", row)
            self.assertNotIn("estimated_field_inf_norm", row)
            self.assertNotIn("true_field_inf_norm", row)
            self.assertGreater(float(np.linalg.norm(estimated_field, ord=np.inf)), 2.0)
        finally:
            shutil.rmtree(fit_root, ignore_errors=True)

    def test_fit_report_latent_diagnostics_reads_legacy_inf_norm_names(self) -> None:
        from utils.t8_parameter_recovery_reporting import latent_diagnostics

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
            self.assertNotIn("estimated_field_frobenius_norm", row)
            self.assertNotIn("true_field_frobenius_norm", row)
            self.assertNotIn("estimated_sv_1", row)
            self.assertNotIn("true_sv_1", row)
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

    def test_beta_mask_pre_s_does_not_change_forward_model(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        masked_context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
            s=1,
            beta_mask_pre_s=True,
        )
        unmasked_context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
            s=1,
            beta_mask_pre_s=False,
        )

        h_masked = _compute_h_x(
            np.zeros_like(x),
            {"beta": 2.0, "xi": 0.0, "eta": 0.0},
            masked_context,
        )
        h_unmasked = _compute_h_x(
            np.zeros_like(x),
            {"beta": 2.0, "xi": 0.0, "eta": 0.0},
            unmasked_context,
        )

        np.testing.assert_allclose(h_masked, h_unmasked)

    def test_beta_mask_pre_s_only_changes_beta_gradient(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        masked_context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
            s=1,
            beta_mask_pre_s=True,
        )
        unmasked_context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
            s=1,
            beta_mask_pre_s=False,
        )

        masked_loss, masked_residual, masked_scalar_grad = _evaluate_full_field_loss(
            np.zeros_like(x),
            masked_context,
            scalar_values={"beta": 0.75, "xi": 0.5, "eta": -0.25},
        )
        unmasked_loss, unmasked_residual, unmasked_scalar_grad = _evaluate_full_field_loss(
            np.zeros_like(x),
            unmasked_context,
            scalar_values={"beta": 0.75, "xi": 0.5, "eta": -0.25},
        )

        self.assertAlmostEqual(masked_loss, unmasked_loss, places=12)
        np.testing.assert_allclose(masked_residual, unmasked_residual)
        np.testing.assert_allclose(masked_scalar_grad[1:], unmasked_scalar_grad[1:])
        expected_beta_gradient = float(
            (
                masked_residual
                * z
                * np.asarray(masked_context.beta_update_mask, dtype=float)
            ).sum()
            / masked_context.beta_outcome_size
        )
        self.assertAlmostEqual(masked_scalar_grad[0], expected_beta_gradient, places=12)
        self.assertNotAlmostEqual(masked_scalar_grad[0], unmasked_scalar_grad[0], places=12)

    def test_beta_gradient_mask_intersects_loss_mask(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=float)
        z = np.array([[-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        loss_mask = np.array(
            [[True, False], [True, True], [False, True]],
            dtype=bool,
        )
        masked_context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
            s=1,
            beta_mask_pre_s=True,
            loss_mask=loss_mask,
        )
        unmasked_context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect(x, gamma),
            {},
            s=1,
            beta_mask_pre_s=False,
            loss_mask=loss_mask,
        )

        masked_loss, masked_residual, masked_scalar_grad = _evaluate_full_field_loss(
            np.zeros_like(x),
            masked_context,
            scalar_values={"beta": 0.75, "xi": 0.5, "eta": -0.25},
        )
        unmasked_loss, unmasked_residual, unmasked_scalar_grad = _evaluate_full_field_loss(
            np.zeros_like(x),
            unmasked_context,
            scalar_values={"beta": 0.75, "xi": 0.5, "eta": -0.25},
        )

        expected_beta_mask = np.asarray(loss_mask, dtype=bool).copy()
        expected_beta_mask[:1, :] = False
        self.assertAlmostEqual(masked_loss, unmasked_loss, places=12)
        np.testing.assert_allclose(masked_residual, unmasked_residual)
        np.testing.assert_array_equal(masked_context.beta_update_mask, expected_beta_mask)
        self.assertEqual(
            int(masked_context.beta_outcome_size),
            int(np.count_nonzero(expected_beta_mask)),
        )
        self.assertAlmostEqual(
            masked_scalar_grad[0],
            float(
                (
                    masked_residual
                    * z
                    * np.asarray(expected_beta_mask, dtype=float)
                ).sum()
                / np.count_nonzero(expected_beta_mask)
            ),
            places=12,
        )
        np.testing.assert_allclose(masked_scalar_grad[1:], unmasked_scalar_grad[1:])

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

    def test_masked_pseudo_nll_matches_manual_subset_average(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[0.5, -0.25], [0.75, 0.1]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=0,
            optimizer_mode="nuclear_norm",
        )
        theta = np.array([0.1, -0.2, 0.05, 0.15, 0.3, -0.25, 0.2], dtype=float)
        loss_mask = np.array([[True, False], [False, True]], dtype=bool)
        interaction_effect_x = interaction_effect(x, gamma)

        masked_loss, masked_grad = pseudo_nll(
            x=x,
            z=z,
            theta=theta,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params={},
            loss_mask=loss_mask,
        )

        field_matrix = theta[:4].reshape(2, 2)
        context = _build_fit_eval_context(
            x,
            z,
            x_0,
            interaction_effect_x,
            {},
            loss_mask=loss_mask,
        )
        kernel_loss, residual, scalar_grad = _evaluate_full_field_loss(
            field_matrix,
            context,
            free_scalar_values=theta[4:],
        )
        kernel_grad = np.concatenate(
            [(residual / np.count_nonzero(loss_mask)).reshape(-1), scalar_grad]
        )
        h_x = _compute_h_x(
            field_matrix,
            {"beta": theta[4], "xi": theta[5], "eta": theta[6]},
            context,
        )
        expected_loss = float(
            (
                np.logaddexp(h_x, -h_x) - x * h_x
            )[loss_mask].mean()
        )

        self.assertAlmostEqual(masked_loss, expected_loss, places=12)
        self.assertAlmostEqual(kernel_loss, expected_loss, places=12)
        self.assertTrue(np.allclose(masked_grad, kernel_grad))

    def test_evaluate_mple_loss_from_parts_matches_masked_pseudo_nll(self) -> None:
        x = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        z = np.array([[0.5, -0.25], [0.75, 0.1]], dtype=float)
        x_0 = np.array([1.0, -1.0], dtype=float)
        gamma = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=0,
            optimizer_mode="nuclear_norm",
        )
        theta = np.array([0.1, -0.2, 0.05, 0.15, 0.3, -0.25, 0.2], dtype=float)
        loss_mask = np.array([[True, False], [False, True]], dtype=bool)
        interaction_effect_x = interaction_effect(x, gamma)
        ref_loss, _ = pseudo_nll(
            x=x,
            z=z,
            theta=theta,
            x_0=x_0,
            artifacts=artifacts,
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params={},
            loss_mask=loss_mask,
        )
        helper_loss = evaluate_mple_loss_from_parts(
            x=x,
            z=z,
            x_0=x_0,
            field_matrix=theta[:4].reshape(2, 2),
            beta=theta[4],
            xi=theta[5],
            eta=theta[6],
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params={},
            loss_mask=loss_mask,
        )
        self.assertAlmostEqual(helper_loss, ref_loss, places=12)

    def test_fit_mple_rejects_beta_masking_for_deprecated_optimizers(self) -> None:
        x = np.ones((2, 2), dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.zeros(2, dtype=float)
        gamma = np.zeros((2, 2), dtype=float)
        for optimizer_mode, latent_rank in [
            ("no_external_field", 0),
            ("nuclear_norm", 0),
            ("concurrent_latent_rank", 1),
            ("exact_rank_manifold", 1),
        ]:
            artifacts = ModelArtifacts(
                gamma_matrix=gamma,
                t_steps=2,
                latent_rank=latent_rank,
                optimizer_mode=optimizer_mode,
            )
            with self.subTest(optimizer_mode=optimizer_mode):
                with self.assertRaisesRegex(
                    ValueError,
                    "beta-gradient-only masking is only supported",
                ):
                    fit_mple(
                        x,
                        z,
                        x_0=x_0,
                        s=1,
                        param_names=parameter_names(artifacts),
                        artifacts=artifacts,
                        interaction_effect_x=interaction_effect(x, gamma),
                        steps=4,
                        tol=1.0e-8,
                        seed=0,
                        verbose_every=0,
                        beta_mask_pre_s=True,
                    )

    def test_alternating_low_rank_supports_beta_gradient_masking(self) -> None:
        x = np.ones((2, 2), dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.zeros(2, dtype=float)
        gamma = np.zeros((2, 2), dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
            optimizer_mode="alternating_latent_rank",
        )
        fixed_scalars = {"xi": 0.0, "eta": 0.0}
        param_keys = parameter_names(
            artifacts,
            fixed_scalar_params=fixed_scalars,
        )
        common_kwargs = {
            "x": x,
            "z": z,
            "x_0": x_0,
            "s": 1,
            "param_names": param_keys,
            "artifacts": artifacts,
            "interaction_effect_x": interaction_effect(x, gamma),
            "steps": 8,
            "tol": 1.0e-8,
            "seed": 0,
            "verbose_every": 0,
            "fixed_scalar_params": fixed_scalars,
            "n_starts": 1,
            "lambda_uv_ridge": 0.0,
        }

        theta_unmasked, history_unmasked, result_unmasked = fit_mple(**common_kwargs)
        theta_masked, history_masked, result_masked = fit_mple(
            **common_kwargs,
            beta_mask_pre_s=True,
        )
        beta_unmasked = float(
            unpack_theta(
                theta_unmasked,
                artifacts,
                fixed_scalar_params=fixed_scalars,
            )["beta"]
        )
        beta_masked = float(
            unpack_theta(
                theta_masked,
                artifacts,
                fixed_scalar_params=fixed_scalars,
            )["beta"]
        )

        self.assertTrue(np.isfinite(history_unmasked[-1]))
        self.assertTrue(np.isfinite(history_masked[-1]))
        self.assertTrue(np.isfinite(float(result_masked["final_penalized_objective"])))
        self.assertGreater(beta_masked, beta_unmasked)

    def test_alternating_low_rank_beta_gradient_masking_requires_eligible_beta_updates(
        self,
    ) -> None:
        x = np.ones((2, 2), dtype=float)
        z = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=float)
        x_0 = np.zeros(2, dtype=float)
        gamma = np.zeros((2, 2), dtype=float)
        artifacts = ModelArtifacts(
            gamma_matrix=gamma,
            t_steps=2,
            latent_rank=1,
            optimizer_mode="alternating_latent_rank",
        )

        with self.assertRaisesRegex(
            ValueError,
            "removed every eligible beta observation",
        ):
            fit_mple(
                x,
                z,
                x_0=x_0,
                s=2,
                param_names=parameter_names(artifacts),
                artifacts=artifacts,
                interaction_effect_x=interaction_effect(x, gamma),
                steps=4,
                tol=1.0e-8,
                seed=0,
                verbose_every=0,
                beta_mask_pre_s=True,
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

        fit_config = build_fit_config(variant, {"N": 4, "T": 3, "s": 1, "e": 3})
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

        fit_config = build_fit_config(variant, {"N": 4, "T": 3, "s": 1, "e": 3})
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

        fit_config = build_fit_config(variant, {"N": 4, "T": 3, "s": 1, "e": 3})
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
            build_fit_config(variant, {"N": 4, "T": 3, "s": 1, "e": 3})


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
                    "    scalars:",
                    "      beta: 0.2",
                    "      xi: 0.1",
                    "      eta: 0.05",
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
        self.assertIn("estimated_field_frobenius_norm", rows[0])
        self.assertIn("true_field_frobenius_norm", rows[0])
        self.assertIn("estimated_field_nuclear_norm", rows[0])
        self.assertIn("true_field_nuclear_norm", rows[0])
        self.assertIn("estimated_singular_value_count", rows[0])
        self.assertIn("true_singular_value_count", rows[0])
        self.assertIn("estimated_sv_1", rows[0])
        self.assertIn("true_sv_1", rows[0])
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
        self.assertIn("estimated_field_frobenius_norm", winner_rows[0])
        self.assertIn("true_field_frobenius_norm", winner_rows[0])
        self.assertIn("estimated_sv_1", winner_rows[0])
        self.assertIn("true_sv_1", winner_rows[0])
        self.assertEqual(
            Path(fit_manifest), self.root / "generated" / "fit_manifest.csv"
        )


class PipelineStageRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / f".tsr_{uuid.uuid4().hex[:6]}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_fixed_generation_artifacts(
        self,
        name: str,
        *,
        gamma_matrix: np.ndarray | None = None,
        z: np.ndarray | None = None,
        z_0: np.ndarray | None = None,
    ) -> dict[str, str]:
        artifact_root = self.root / "artifacts" / name
        artifact_root.mkdir(parents=True, exist_ok=True)
        result: dict[str, str] = {"artifact_dir": artifact_root.as_posix()}
        if gamma_matrix is not None:
            gamma_path = artifact_root / "gamma_matrix.npy"
            np.save(gamma_path, np.asarray(gamma_matrix, dtype=float))
            result["gamma_path"] = gamma_path.as_posix()
        if z is not None:
            z_array = np.asarray(z, dtype=float)
            z0_array = (
                -np.ones(z_array.shape[1], dtype=float)
                if z_0 is None
                else np.asarray(z_0, dtype=float)
            )
            panel_path = artifact_root / "panel_data.npz"
            z0_path = artifact_root / "z_0.npy"
            np.savez(panel_path, x=np.zeros_like(z_array), z=z_array)
            np.save(z0_path, z0_array)
            result["panel_path"] = panel_path.as_posix()
            result["z0_path"] = z0_path.as_posix()
        return result

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
                    "dimensions": {"N": 6, "T": 4},
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
                    "field_mode": "random_low_rank",
                    "field_params": {},
                    "scalars": {
                        "beta": 0.2,
                        "xi": 0.1,
                        "eta": 0.05,
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

    def _write_cv_spec(self, searches: list[dict[str, object] | str]) -> Path:
        normalized_searches: list[dict[str, object]] = []
        for search in searches:
            if isinstance(search, str):
                normalized_searches.append({"name": search})
            else:
                normalized_searches.append(dict(search))
        cv_spec_path = self.root / "cv_spec.yaml"
        spec = {
            "base": {
                "cv_root_name": "cv_runs",
                "cv_manifest_path": f"{self.root.as_posix()}/generated/cv_manifest.csv",
                "validation_root_name": "validation_runs",
                "validation_manifest_path": f"{self.root.as_posix()}/generated/validation_manifest.csv",
                "optimizer": {"steps": 3, "tol": 1.0e-6, "seed": 0, "n_starts": 1},
                "latent_rank": 0,
                "optimizer_mode": "no_external_field",
                "lambda_nuclear": 0.0,
                "lambda_frobenius": 0.0,
                "lambda_uv_ridge": 0.0,
                "validation_sampling": {"num_samples": 2, "gibbs_sweeps": 1, "seed": 0},
                "estimation": {"fixed_scalar_params": {}},
            },
            "searches": normalized_searches,
        }
        OmegaConf.save(OmegaConf.create(spec), cv_spec_path)
        return cv_spec_path

    def _write_fit_manifest_rows(self, rows: list[dict[str, object]]) -> Path:
        manifest_path = self.root / "fit_manifest.csv"
        write_csv_rows(manifest_path, rows)
        return manifest_path

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

    def _write_fake_pixi(self) -> tuple[Path, Path]:
        fake_pixi_path = self.root / "pixi"
        fake_log_path = self.root / "fake_pixi_log.txt"
        fake_pixi_path.write_text(
            "\n".join(
                [
                    "#!/bin/bash",
                    "set -euo pipefail",
                    'printf "%s\\n" "$*" >> "${FAKE_PIXI_LOG}"',
                ]
            ),
            encoding="utf-8",
        )
        fake_pixi_path.chmod(0o755)
        return fake_pixi_path, fake_log_path

    def _write_env_logging_sbatch(self) -> tuple[Path, Path, Path]:
        fake_sbatch_path = self.root / "fake_sbatch_env.sh"
        fake_counter_path = self.root / "fake_sbatch_env_counter.txt"
        fake_log_path = self.root / "fake_sbatch_env_log.txt"
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
                    'printf "env:CV_CONTINUE=%s|" "${CV_CONTINUE:-}" >> "${FAKE_SBATCH_LOG}"',
                    'printf "<%s>" "$@" >> "${FAKE_SBATCH_LOG}"',
                    'printf "\\n" >> "${FAKE_SBATCH_LOG}"',
                    'printf "%s\\n" "job${count}"',
                ]
            ),
            encoding="utf-8",
        )
        fake_sbatch_path.chmod(0o755)
        return fake_sbatch_path, fake_counter_path, fake_log_path

    @staticmethod
    def _deterministic_fake_pymetis():
        class FakePyMetis:
            @staticmethod
            def part_graph(nparts, adjacency=None, recursive=None, contiguous=None):
                n = len(adjacency or [])
                if n <= 0:
                    return 0, []
                return 0, [min((index * int(nparts)) // n, int(nparts) - 1) for index in range(n)]

        return FakePyMetis()

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
        row = run_generation_request(generation_spec_path, "exp_a", overwrite=True)

        self.assertTrue((self.root / "generated" / "exp_a" / "panel_data.npz").exists())
        self.assertFalse((self.root / "generated" / "exp_b").exists())
        with np.load(
            self.root / "generated" / "exp_a" / "field_artifacts.npz",
            allow_pickle=False,
        ) as data:
            expected_latent_rank = int(np.asarray(data["latent_rank"]).item())
        self.assertEqual(int(row["latent_rank"]), expected_latent_rank)
        self.assertEqual(row["experiment_name"], "exp_a")

    def test_run_generation_request_fixed_intervention_resolves_null_dimensions(self) -> None:
        fixed_artifacts = self._write_fixed_generation_artifacts(
            "fixed_z_only",
            z=np.array(
                [
                    [1.0, -1.0, 1.0, -1.0, 1.0],
                    [-1.0, 1.0, -1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0, -1.0, 1.0],
                ],
                dtype=float,
            ),
        )
        generation_spec_path = self._write_generation_spec(
            [
                {
                    "name": "exp_fixed_z_null_dims",
                    "dimensions": {"N": None, "T": None},
                    "intervention": {
                        "source": "fixed_artifact",
                        "artifact": {
                            "panel_path": fixed_artifacts["panel_path"],
                            "z0_path": fixed_artifacts["z0_path"],
                            "artifact_dir": fixed_artifacts["artifact_dir"],
                            "shared_panel_dir": None,
                            "outcome_code": None,
                            "intervention_code": None,
                            "lag_code": None,
                            "trim_scope": None,
                        },
                    },
                }
            ]
        )

        row = run_generation_request(
            generation_spec_path, "exp_fixed_z_null_dims", overwrite=True
        )

        realized_config = OmegaConf.load(
            self.root
            / "generated"
            / "exp_fixed_z_null_dims"
            / "generation_realized_config.yaml"
        )
        self.assertEqual(int(realized_config.global_params.N), 5)
        self.assertEqual(int(realized_config.global_params.T), 3)
        self.assertEqual(
            set(OmegaConf.to_container(realized_config.estimation_params, resolve=True)),
            {"beta", "xi", "eta"},
        )
        self.assertEqual(int(row["N"]), 5)
        self.assertEqual(int(row["T"]), 3)

    def test_run_generation_request_fixed_graph_resolves_null_n(self) -> None:
        fixed_artifacts = self._write_fixed_generation_artifacts(
            "fixed_gamma_only",
            gamma_matrix=np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
        )
        generation_spec_path = self._write_generation_spec(
            [
                {
                    "name": "exp_fixed_gamma_null_n",
                    "dimensions": {"N": None, "T": 3},
                    "graph": {
                        "source": "fixed_artifact",
                        "artifact": {
                            "gamma_path": fixed_artifacts["gamma_path"],
                            "node_index_path": None,
                            "artifact_dir": fixed_artifacts["artifact_dir"],
                            "network_name": None,
                            "trim_scope": None,
                        },
                    },
                }
            ]
        )

        row = run_generation_request(
            generation_spec_path, "exp_fixed_gamma_null_n", overwrite=True
        )

        realized_config = OmegaConf.load(
            self.root
            / "generated"
            / "exp_fixed_gamma_null_n"
            / "generation_realized_config.yaml"
        )
        self.assertEqual(int(realized_config.global_params.N), 5)
        self.assertEqual(int(realized_config.global_params.T), 3)
        self.assertEqual(int(row["N"]), 5)
        self.assertEqual(int(row["T"]), 3)

    def test_run_generation_request_generated_graph_requires_resolved_n(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_missing_n", "dimensions": {"N": None, "T": 4}}]
        )

        with self.assertRaisesRegex(
            ValueError,
            "global_params.N must be resolved before generating a non-fixed graph",
        ):
            run_generation_request(generation_spec_path, "exp_missing_n", overwrite=True)

    def test_run_generation_request_generated_intervention_requires_resolved_t(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_missing_t", "dimensions": {"N": 6, "T": None}}]
        )

        with self.assertRaisesRegex(
            ValueError,
            "global_params.T must be resolved before generation",
        ):
            run_generation_request(generation_spec_path, "exp_missing_t", overwrite=True)

    def test_run_generation_request_conflicting_fixed_sources_raise(self) -> None:
        fixed_graph = self._write_fixed_generation_artifacts(
            "conflicting_fixed_graph",
            gamma_matrix=np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
        )
        fixed_intervention = self._write_fixed_generation_artifacts(
            "conflicting_fixed_z",
            z=np.array(
                [
                    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                    [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                    [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                ],
                dtype=float,
            ),
        )
        generation_spec_path = self._write_generation_spec(
            [
                {
                    "name": "exp_conflicting_fixed_sources",
                    "dimensions": {"N": None, "T": None},
                    "graph": {
                        "source": "fixed_artifact",
                        "artifact": {
                            "gamma_path": fixed_graph["gamma_path"],
                            "node_index_path": None,
                            "artifact_dir": fixed_graph["artifact_dir"],
                            "network_name": None,
                            "trim_scope": None,
                        },
                    },
                    "intervention": {
                        "source": "fixed_artifact",
                        "artifact": {
                            "panel_path": fixed_intervention["panel_path"],
                            "z0_path": fixed_intervention["z0_path"],
                            "artifact_dir": fixed_intervention["artifact_dir"],
                            "shared_panel_dir": None,
                            "outcome_code": None,
                            "intervention_code": None,
                            "lag_code": None,
                            "trim_scope": None,
                        },
                    },
                }
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not match configured N=6",
        ):
            run_generation_request(
                generation_spec_path,
                "exp_conflicting_fixed_sources",
                overwrite=True,
            )

    def test_refresh_generation_manifest_rebuilds_manifest_from_outputs(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [
                {
                    "name": "exp_rank_2",
                    "truth": {
                        "field_mode": "random_low_rank",
                        "field_params": {"singular_values": [1.0, 0.7]},
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

    def test_write_train_fit_requests_writes_one_row_per_experiment_for_train_cv(self) -> None:
        generation_spec_path = self._write_generation_spec([{"name": "exp_a", "dimensions": {"T": 9}}])
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest, overwrite=True)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "mask_grid",
                    "grid": {"estimation": {"beta_mask_pre_s": [False]}},
                }
            ]
        )
        cv_runner.run_cv_folds(generation_manifest, cv_spec_path, overwrite=True)

        request_path = write_train_fit_requests(
            generation_manifest,
            cv_spec_path,
            "mask_grid",
        )

        rows = read_csv_manifest(request_path)
        self.assertEqual(request_path.name, "train_fit_requests__mask_grid.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["search_slug"], "mask_grid")
        self.assertEqual(rows[0]["split_kind"], "train_cv")
        self.assertEqual(int(rows[0]["num_folds"]), 5)
        self.assertTrue(rows[0]["best_candidate_path"].endswith("best_candidate.yaml"))
        self.assertIn(
            str(
                self.root
                / "generated"
                / "exp_a"
                / "train_fits"
                / "mask_grid"
                / "train_cv__folds_5"
            ),
            rows[0]["fit_path"],
        )

    def test_write_train_fit_requests_fails_cleanly_without_best_candidate(self) -> None:
        generation_spec_path = self._write_generation_spec([{"name": "exp_a", "dimensions": {"T": 9}}])
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest, overwrite=True)
        cv_spec_path = self._write_cv_spec(
            [{"name": "mask_grid", "grid": {"estimation": {"beta_mask_pre_s": [False]}}}]
        )

        with self.assertRaisesRegex(FileNotFoundError, "Best candidate YAML not found"):
            write_train_fit_requests(generation_manifest, cv_spec_path, "mask_grid")

    def test_run_train_fit_request_uses_outer_training_mask_for_train_cv(self) -> None:
        generation_spec_path = self._write_generation_spec([{"name": "exp_a", "dimensions": {"T": 9}}])
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest, overwrite=True)
        cv_spec_path = self._write_cv_spec(
            [{"name": "mask_grid", "grid": {"estimation": {"beta_mask_pre_s": [False]}}}]
        )
        cv_runner.run_cv_folds(generation_manifest, cv_spec_path, overwrite=True)

        row = run_train_fit_request(
            generation_manifest,
            cv_spec_path,
            "mask_grid",
            "exp_a",
            overwrite=True,
        )

        fit_root = Path(row["fit_path"])
        self.assertTrue((fit_root / "loss_mask.npy").exists())
        self.assertTrue((fit_root / "fit_realized_config.yaml").exists())
        self.assertTrue((fit_root / "fit_metadata.yaml").exists())
        self.assertTrue((fit_root / "mple_summary.csv").exists())

        metadata = load_yaml_mapping(fit_root / "fit_metadata.yaml")
        self.assertEqual(metadata["execution_mode"], "train_fit")
        self.assertEqual(metadata["search_slug"], "mask_grid")
        self.assertEqual(metadata["split_kind"], "train_cv")

        split_artifacts = load_outer_training_split_masks(
            self.root / "generated" / "exp_a",
            split_kind="train_cv",
            num_folds=5,
        )
        expected_mask = np.asarray(split_artifacts["training_mask"], dtype=bool)
        observed_mask = np.load(io_path(fit_root / "loss_mask.npy"))
        np.testing.assert_array_equal(observed_mask, expected_mask)
        self.assertEqual(int(metadata["num_training_slots"]), int(np.count_nonzero(expected_mask)))

    def test_refresh_train_fit_manifest_rebuilds_manifest_and_reports(self) -> None:
        generation_spec_path = self._write_generation_spec([{"name": "exp_a", "dimensions": {"T": 9}}])
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest, overwrite=True)
        cv_spec_path = self._write_cv_spec(
            [{"name": "mask_grid", "grid": {"estimation": {"beta_mask_pre_s": [False]}}}]
        )
        cv_runner.run_cv_folds(generation_manifest, cv_spec_path, overwrite=True)
        run_train_fit_request(
            generation_manifest,
            cv_spec_path,
            "mask_grid",
            "exp_a",
            overwrite=True,
        )

        manifest_path = refresh_train_fit_manifest(
            generation_manifest,
            cv_spec_path,
            "mask_grid",
        )

        rows = read_csv_manifest(manifest_path)
        self.assertEqual(manifest_path.name, "train_fit_manifest__mask_grid.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["execution_mode"], "train_fit")
        self.assertEqual(rows[0]["search_slug"], "mask_grid")
        self.assertTrue(
            (self.root / "generated" / "exp_a" / "train_fit_summary__mask_grid.csv").exists()
        )
        self.assertTrue(
            (self.root / "generated" / "best_train_fit_by_experiment__mask_grid.csv").exists()
        )

    def test_refresh_train_fit_manifest_without_search_slug_combines_all_searches(self) -> None:
        generation_spec_path = self._write_generation_spec([{"name": "exp_a", "dimensions": {"T": 9}}])
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest, overwrite=True)
        cv_spec_path = self._write_cv_spec(
            [
                {"name": "mask_grid", "grid": {"estimation": {"beta_mask_pre_s": [False]}}},
                {"name": "mask_grid_alt", "grid": {"estimation": {"beta_mask_pre_s": [True]}}},
            ]
        )
        cv_runner.run_cv_folds(generation_manifest, cv_spec_path, overwrite=True)
        run_train_fit_request(
            generation_manifest,
            cv_spec_path,
            "mask_grid",
            "exp_a",
            overwrite=True,
        )
        run_train_fit_request(
            generation_manifest,
            cv_spec_path,
            "mask_grid_alt",
            "exp_a",
            overwrite=True,
        )

        manifest_path = refresh_train_fit_manifest(
            generation_manifest,
            cv_spec_path,
        )

        rows = read_csv_manifest(manifest_path)
        self.assertEqual(manifest_path.name, "train_fit_manifest.csv")
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["search_slug"] for row in rows}, {"mask_grid", "mask_grid_alt"})
        self.assertTrue(
            (self.root / "generated" / "exp_a" / "train_fit_summary.csv").exists()
        )
        self.assertTrue(
            (self.root / "generated" / "best_train_fit_by_experiment.csv").exists()
        )

    def test_run_train_fit_request_uses_outer_training_mask_for_test_train_cv(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [
                {
                    "name": "exp_a",
                    "dimensions": {"N": 72, "T": 12},
                    "graph": {"params": {"p": 0.08}},
                }
            ]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        experiment_root = Path(read_csv_manifest(generation_manifest)[0]["experiment_path"]).resolve()
        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=self._deterministic_fake_pymetis(),
        ):
            uscounty_splits.build_validation_test_splits_for_experiment(
                experiment_root,
                outer_num_folds=3,
                test_fold_id=1,
                inner_num_folds=3,
                overwrite=True,
            )
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "outer_mask_grid",
                    "split_kind": "test_train_cv",
                    "num_folds": 3,
                    "outer_num_folds": 3,
                    "test_fold_id": 1,
                    "grid": {"estimation": {"beta_mask_pre_s": [False]}},
                }
            ]
        )
        cv_runner.run_cv_folds(generation_manifest, cv_spec_path, overwrite=True)

        row = run_train_fit_request(
            generation_manifest,
            cv_spec_path,
            "outer_mask_grid",
            "exp_a",
            overwrite=True,
        )

        fit_root = Path(row["fit_path"])
        metadata = load_yaml_mapping(fit_root / "fit_metadata.yaml")
        self.assertEqual(metadata["split_kind"], "test_train_cv")
        self.assertEqual(int(metadata["outer_num_folds"]), 3)
        self.assertEqual(int(metadata["test_fold_id"]), 1)

        split_artifacts = load_outer_training_split_masks(
            experiment_root,
            split_kind="test_train_cv",
            num_folds=3,
            outer_num_folds=3,
            test_fold_id=1,
        )
        expected_mask = np.asarray(split_artifacts["training_mask"], dtype=bool)
        observed_mask = np.load(io_path(fit_root / "loss_mask.npy"))
        np.testing.assert_array_equal(observed_mask, expected_mask)
        self.assertEqual(int(metadata["num_training_slots"]), int(np.count_nonzero(expected_mask)))

    def test_validate_cv_spec_accepts_grid_search(self) -> None:
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "alternating_uv_grid",
                    "optimizer_mode": "alternating_latent_rank",
                    "grid": {
                        "latent_rank": [1, 2],
                        "lambda_uv_ridge": [0.01, 0.1],
                    },
                }
            ]
        )

        validate_cv_spec(cv_spec_path)
        searches = cv_runner._expand_searches(cv_spec_path)
        candidates = cv_runner.expand_search_candidates(searches[0])

        self.assertEqual(len(candidates), 4)

    def test_materialize_fit_root_supports_loss_mask_path(self) -> None:
        generation_spec_path = self._write_generation_spec(["exp_a"])
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        experiment_row = read_csv_manifest(generation_manifest)[0]
        fit_root = self.root / "generated" / "exp_a" / "masked_fit"
        fit_root.mkdir(parents=True, exist_ok=False)
        loss_mask = np.array(
            [[True, False, True, False, True, False]] * 4,
            dtype=bool,
        )
        loss_mask_path = fit_root / "loss_mask.npy"
        np.save(io_path(loss_mask_path), loss_mask)
        variant = {
            "name": "rank_0_masked",
            "slug": "rank_0_masked",
            "optimizer": {"steps": 5, "tol": 1.0e-6, "seed": 0},
            "optimizer_mode": "no_external_field",
            "latent_rank": 0,
            "lambda_nuclear": 0.0,
            "lambda_frobenius": 0.0,
            "lambda_uv_ridge": 0.0,
            "estimation": {"fixed_scalar_params": {}},
        }
        materialize_fit_root(
            experiment_row,
            variant,
            fit_root,
            extra_input_artifacts={"loss_mask_path": str(loss_mask_path.resolve())},
        )

        execute_fit_root(fit_root)
        config = OmegaConf.load(fit_root / "fit_realized_config.yaml")
        self.assertEqual(
            str(config.input_artifacts.loss_mask_path),
            str(loss_mask_path.resolve()),
        )
        self.assertTrue((fit_root / "mple_summary.csv").exists())

    def test_run_cv_folds_writes_requests_manifest_and_scores(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "nefg",
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0, 1]}},
                }
            ]
        )

        manifest_path = cv_runner.run_cv_folds(
            generation_manifest,
            cv_spec_path,
            overwrite=True,
        )

        manifest_rows = read_csv_manifest(manifest_path)
        self.assertEqual(len(manifest_rows), 1)
        self.assertEqual(manifest_rows[0]["status"], "completed")
        self.assertEqual(manifest_rows[0]["search_slug"], "nefg")

        requests_path = cv_runner.cv_requests_path_for_spec(cv_spec_path)
        request_rows = read_csv_manifest(requests_path)
        self.assertEqual(len(request_rows), 10)

        output_root = (
            self.root
            / "generated"
            / "exp_a"
            / "cv_runs"
            / "nefg"
        )
        candidate_rows = read_csv_manifest(output_root / "candidate_grid.csv")
        fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        score_rows = read_csv_manifest(output_root / "candidate_scores.csv")
        best_candidate = OmegaConf.load(output_root / "best_candidate.yaml")

        self.assertEqual(len(candidate_rows), 2)
        self.assertEqual(len(fold_rows), 10)
        self.assertEqual(len(score_rows), 2)
        self.assertEqual(str(best_candidate.search_slug), "nefg")
        self.assertIn("weighted_mean_validation_brier_score", manifest_rows[0])
        self.assertIn("mean_fold_validation_brier_score", manifest_rows[0])
        self.assertIn("weighted_mean_validation_ece", manifest_rows[0])
        self.assertIn("mean_fold_validation_ece", manifest_rows[0])
        self.assertIn("weighted_mean_validation_mean_magnetization_abs_diff", manifest_rows[0])
        self.assertIn("mean_fold_validation_mean_magnetization_abs_diff", manifest_rows[0])
        self.assertIn("weighted_mean_post_s_validation_loss", manifest_rows[0])
        self.assertIn("mean_fold_post_s_validation_loss", manifest_rows[0])
        self.assertIn("weighted_mean_post_s_validation_brier_score", manifest_rows[0])
        self.assertIn("mean_fold_post_s_validation_brier_score", manifest_rows[0])
        self.assertIn("standard_error_fold_post_s_validation_brier_score", manifest_rows[0])
        self.assertIn("weighted_mean_post_s_validation_ece", manifest_rows[0])
        self.assertIn("mean_fold_post_s_validation_ece", manifest_rows[0])
        self.assertIn(
            "weighted_mean_post_s_validation_mean_magnetization_abs_diff",
            manifest_rows[0],
        )
        self.assertIn(
            "mean_fold_post_s_validation_mean_magnetization_abs_diff",
            manifest_rows[0],
        )
        self.assertIn(
            "standard_error_fold_post_s_validation_mean_magnetization_abs_diff",
            manifest_rows[0],
        )
        self.assertIn("weighted_mean_validation_brier_score", best_candidate)
        self.assertIn("mean_fold_validation_brier_score", best_candidate)
        self.assertIn("weighted_mean_validation_ece", best_candidate)
        self.assertIn("mean_fold_validation_ece", best_candidate)
        self.assertIn("weighted_mean_validation_mean_magnetization_abs_diff", best_candidate)
        self.assertIn("mean_fold_validation_mean_magnetization_abs_diff", best_candidate)
        self.assertIn("weighted_mean_post_s_validation_loss", best_candidate)
        self.assertIn("mean_fold_post_s_validation_loss", best_candidate)
        self.assertIn("weighted_mean_post_s_validation_brier_score", best_candidate)
        self.assertIn("mean_fold_post_s_validation_brier_score", best_candidate)
        self.assertIn("standard_error_fold_post_s_validation_brier_score", best_candidate)
        self.assertIn("weighted_mean_post_s_validation_ece", best_candidate)
        self.assertIn("mean_fold_post_s_validation_ece", best_candidate)
        self.assertIn(
            "weighted_mean_post_s_validation_mean_magnetization_abs_diff",
            best_candidate,
        )
        self.assertIn(
            "mean_fold_post_s_validation_mean_magnetization_abs_diff",
            best_candidate,
        )
        self.assertIn(
            "standard_error_fold_post_s_validation_mean_magnetization_abs_diff",
            best_candidate,
        )

        completed_rows = [row for row in fold_rows if row["status"] == "completed"]
        self.assertEqual(len(completed_rows), 10)
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in completed_rows:
            grouped.setdefault(row["candidate_slug"], []).append(row)
        for row in score_rows:
            if row["status"] != "completed":
                continue
            candidate_rows_group = grouped[row["candidate_slug"]]
            for fold_row in candidate_rows_group:
                self.assertIn("validation_brier_score", fold_row)
                self.assertIn("validation_ece", fold_row)
                self.assertIn("validation_mean_magnetization_abs_diff", fold_row)
                self.assertIn("validation_observed_mean_magnetization", fold_row)
                self.assertIn("validation_sampled_mean_magnetization_mean", fold_row)
                self.assertIn("num_post_s_validation_slots", fold_row)
                self.assertIn("post_s_validation_loss", fold_row)
                self.assertIn("post_s_validation_brier_score", fold_row)
                self.assertIn("post_s_validation_ece", fold_row)
                self.assertIn("post_s_validation_mean_magnetization_abs_diff", fold_row)
                self.assertIn("post_s_validation_observed_mean_magnetization", fold_row)
                self.assertIn("post_s_validation_sampled_mean_magnetization_mean", fold_row)
            weighted = sum(
                float(fold_row["validation_loss"]) * int(fold_row["num_validation_slots"])
                for fold_row in candidate_rows_group
            ) / sum(int(fold_row["num_validation_slots"]) for fold_row in candidate_rows_group)
            weighted_brier = sum(
                float(fold_row["validation_brier_score"])
                * int(fold_row["num_validation_slots"])
                for fold_row in candidate_rows_group
            ) / sum(int(fold_row["num_validation_slots"]) for fold_row in candidate_rows_group)
            mean_fold_brier = sum(
                float(fold_row["validation_brier_score"])
                for fold_row in candidate_rows_group
            ) / len(candidate_rows_group)
            weighted_ece = sum(
                float(fold_row["validation_ece"])
                * int(fold_row["num_validation_slots"])
                for fold_row in candidate_rows_group
            ) / sum(int(fold_row["num_validation_slots"]) for fold_row in candidate_rows_group)
            weighted_mag_diff = sum(
                float(fold_row["validation_mean_magnetization_abs_diff"])
                * int(fold_row["num_validation_slots"])
                for fold_row in candidate_rows_group
            ) / sum(int(fold_row["num_validation_slots"]) for fold_row in candidate_rows_group)
            mean_fold_ece = sum(
                float(fold_row["validation_ece"])
                for fold_row in candidate_rows_group
            ) / len(candidate_rows_group)
            mean_fold_mag_diff = sum(
                float(fold_row["validation_mean_magnetization_abs_diff"])
                for fold_row in candidate_rows_group
            ) / len(candidate_rows_group)
            post_s_rows = [
                fold_row
                for fold_row in candidate_rows_group
                if int(fold_row["num_post_s_validation_slots"]) > 0
            ]
            self.assertAlmostEqual(
                float(row["weighted_mean_validation_loss"]),
                weighted,
                places=12,
            )
            self.assertAlmostEqual(
                float(row["weighted_mean_validation_brier_score"]),
                weighted_brier,
                places=12,
            )
            self.assertAlmostEqual(
                float(row["mean_fold_validation_brier_score"]),
                mean_fold_brier,
                places=12,
            )
            self.assertAlmostEqual(
                float(row["weighted_mean_validation_ece"]),
                weighted_ece,
                places=12,
            )
            self.assertAlmostEqual(
                float(row["mean_fold_validation_ece"]),
                mean_fold_ece,
                places=12,
            )
            self.assertAlmostEqual(
                float(row["weighted_mean_validation_mean_magnetization_abs_diff"]),
                weighted_mag_diff,
                places=12,
            )
            self.assertAlmostEqual(
                float(row["mean_fold_validation_mean_magnetization_abs_diff"]),
                mean_fold_mag_diff,
                places=12,
            )
            if post_s_rows:
                weighted_post_s_loss = sum(
                    float(fold_row["post_s_validation_loss"])
                    * int(fold_row["num_post_s_validation_slots"])
                    for fold_row in post_s_rows
                ) / sum(
                    int(fold_row["num_post_s_validation_slots"])
                    for fold_row in post_s_rows
                )
                mean_fold_post_s_loss = sum(
                    float(fold_row["post_s_validation_loss"])
                    for fold_row in post_s_rows
                ) / len(post_s_rows)
                weighted_post_s_brier = sum(
                    float(fold_row["post_s_validation_brier_score"])
                    * int(fold_row["num_post_s_validation_slots"])
                    for fold_row in post_s_rows
                ) / sum(
                    int(fold_row["num_post_s_validation_slots"])
                    for fold_row in post_s_rows
                )
                mean_fold_post_s_brier = sum(
                    float(fold_row["post_s_validation_brier_score"])
                    for fold_row in post_s_rows
                ) / len(post_s_rows)
                post_s_brier_values = [
                    float(fold_row["post_s_validation_brier_score"])
                    for fold_row in post_s_rows
                ]
                weighted_post_s_ece = sum(
                    float(fold_row["post_s_validation_ece"])
                    * int(fold_row["num_post_s_validation_slots"])
                    for fold_row in post_s_rows
                ) / sum(
                    int(fold_row["num_post_s_validation_slots"])
                    for fold_row in post_s_rows
                )
                weighted_post_s_mag_diff = sum(
                    float(fold_row["post_s_validation_mean_magnetization_abs_diff"])
                    * int(fold_row["num_post_s_validation_slots"])
                    for fold_row in post_s_rows
                ) / sum(
                    int(fold_row["num_post_s_validation_slots"])
                    for fold_row in post_s_rows
                )
                mean_fold_post_s_ece = sum(
                    float(fold_row["post_s_validation_ece"])
                    for fold_row in post_s_rows
                ) / len(post_s_rows)
                mean_fold_post_s_mag_diff = sum(
                    float(fold_row["post_s_validation_mean_magnetization_abs_diff"])
                    for fold_row in post_s_rows
                ) / len(post_s_rows)
                post_s_mag_diff_values = [
                    float(fold_row["post_s_validation_mean_magnetization_abs_diff"])
                    for fold_row in post_s_rows
                ]
                if len(post_s_brier_values) <= 1:
                    se_post_s_brier = 0.0
                else:
                    se_post_s_brier = float(
                        np.std(post_s_brier_values, ddof=1)
                        / np.sqrt(len(post_s_brier_values))
                    )
                if len(post_s_mag_diff_values) <= 1:
                    se_post_s_mag_diff = 0.0
                else:
                    se_post_s_mag_diff = float(
                        np.std(post_s_mag_diff_values, ddof=1)
                        / np.sqrt(len(post_s_mag_diff_values))
                    )
                self.assertAlmostEqual(
                    float(row["weighted_mean_post_s_validation_loss"]),
                    weighted_post_s_loss,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(row["mean_fold_post_s_validation_loss"]),
                    mean_fold_post_s_loss,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(row["weighted_mean_post_s_validation_brier_score"]),
                    weighted_post_s_brier,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(row["mean_fold_post_s_validation_brier_score"]),
                    mean_fold_post_s_brier,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(row["standard_error_fold_post_s_validation_brier_score"]),
                    se_post_s_brier,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(row["weighted_mean_post_s_validation_ece"]),
                    weighted_post_s_ece,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(row["mean_fold_post_s_validation_ece"]),
                    mean_fold_post_s_ece,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(row["weighted_mean_post_s_validation_mean_magnetization_abs_diff"]),
                    weighted_post_s_mag_diff,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(row["mean_fold_post_s_validation_mean_magnetization_abs_diff"]),
                    mean_fold_post_s_mag_diff,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(
                        row[
                            "standard_error_fold_post_s_validation_mean_magnetization_abs_diff"
                        ]
                    ),
                    se_post_s_mag_diff,
                    places=12,
                )
            else:
                self.assertEqual(row["weighted_mean_post_s_validation_loss"], "")
                self.assertEqual(row["mean_fold_post_s_validation_loss"], "")
                self.assertEqual(row["weighted_mean_post_s_validation_brier_score"], "")
                self.assertEqual(row["mean_fold_post_s_validation_brier_score"], "")
                self.assertEqual(row["standard_error_fold_post_s_validation_brier_score"], "")
                self.assertEqual(row["weighted_mean_post_s_validation_ece"], "")
                self.assertEqual(row["mean_fold_post_s_validation_ece"], "")
                self.assertEqual(
                    row["weighted_mean_post_s_validation_mean_magnetization_abs_diff"],
                    "",
                )
                self.assertEqual(
                    row["mean_fold_post_s_validation_mean_magnetization_abs_diff"],
                    "",
                )
                self.assertEqual(
                    row["standard_error_fold_post_s_validation_mean_magnetization_abs_diff"],
                    "",
                )

    def test_run_cv_folds_persists_fold_scores_before_interruption(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "pps",
                    "optimizer_mode": "no_external_field",
                    "grid": {
                        "estimation": {
                            "beta_mask_pre_s": [False],
                        }
                    },
                }
            ]
        )

        original_evaluator = cv_runner._evaluate_and_store_fold_metrics
        call_count = {"value": 0}

        def interrupting_evaluator(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 2:
                raise KeyboardInterrupt("simulated interruption")
            return original_evaluator(*args, **kwargs)

        with mock.patch.object(
            cv_runner,
            "_evaluate_and_store_fold_metrics",
            side_effect=interrupting_evaluator,
        ):
            with self.assertRaisesRegex(KeyboardInterrupt, "simulated interruption"):
                cv_runner.run_cv_folds(
                    generation_manifest,
                    cv_spec_path,
                    overwrite=True,
                )

        output_root = (
            self.root
            / "generated"
            / "exp_a"
            / "cv_runs"
            / "pps"
        )
        fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        self.assertEqual(len(fold_rows), 1)
        self.assertEqual(fold_rows[0]["status"], "completed")
        self.assertFalse((output_root / "candidate_scores.csv").exists())
        self.assertFalse((output_root / "best_candidate.yaml").exists())

    def test_run_cv_folds_continue_reuses_matching_completed_fold(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "rvs",
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0]}},
                }
            ]
        )

        cv_runner.run_cv_folds(
            generation_manifest,
            cv_spec_path,
            execution_mode="validation",
            overwrite=True,
        )

        with mock.patch.object(
            cv_runner,
            "execute_fit_root",
            side_effect=AssertionError("should not refit matching completed fold"),
        ):
            manifest_path = cv_runner.run_cv_folds(
                generation_manifest,
                cv_spec_path,
                execution_mode="validation",
                continue_mode=True,
            )

        manifest_rows = read_csv_manifest(manifest_path)
        self.assertEqual(len(manifest_rows), 1)
        self.assertEqual(manifest_rows[0]["status"], "completed")
        output_root = (
            self.root
            / "generated"
            / "exp_a"
            / "validation_runs"
            / "rvs"
        )
        fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        self.assertEqual(len(fold_rows), 1)
        self.assertEqual(fold_rows[0]["status"], "completed")

    def test_run_cv_folds_continue_errors_on_completed_fold_mismatch(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "rms",
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0]}},
                }
            ]
        )

        cv_runner.run_cv_folds(
            generation_manifest,
            cv_spec_path,
            execution_mode="validation",
            overwrite=True,
        )

        output_root = (
            self.root
            / "generated"
            / "exp_a"
            / "validation_runs"
            / "rms"
        )
        fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        fit_root = Path(fold_rows[0]["fit_path"])
        metadata = load_yaml_mapping(fit_root / "fit_metadata.yaml")
        metadata["search_slug"] = "mutated_search_slug"
        OmegaConf.save(OmegaConf.create(metadata), fit_root / "fit_metadata.yaml")

        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            cv_runner.run_cv_folds(
                generation_manifest,
                cv_spec_path,
                execution_mode="validation",
                continue_mode=True,
            )

    def test_run_cv_folds_continue_reruns_incomplete_fold(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "ris",
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0]}},
                }
            ]
        )

        cv_runner.run_cv_folds(
            generation_manifest,
            cv_spec_path,
            execution_mode="validation",
            overwrite=True,
        )

        output_root = (
            self.root
            / "generated"
            / "exp_a"
            / "validation_runs"
            / "ris"
        )
        fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        fit_root = Path(fold_rows[0]["fit_path"])
        (fit_root / "mple_summary.csv").unlink()

        original_execute_fit_root = cv_runner.execute_fit_root
        with mock.patch.object(
            cv_runner,
            "execute_fit_root",
            wraps=original_execute_fit_root,
        ) as execute_mock:
            manifest_path = cv_runner.run_cv_folds(
                generation_manifest,
                cv_spec_path,
                execution_mode="validation",
                continue_mode=True,
            )

        self.assertEqual(execute_mock.call_count, 1)
        self.assertTrue((fit_root / "mple_summary.csv").exists())
        manifest_rows = read_csv_manifest(manifest_path)
        self.assertEqual(manifest_rows[0]["status"], "completed")

    def test_refresh_cv_scores_from_requests_rebuilds_score_artifacts(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "nefg",
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0, 1]}},
                }
            ]
        )

        manifest_path = cv_runner.run_cv_folds(
            generation_manifest,
            cv_spec_path,
            overwrite=True,
        )
        requests_path = cv_runner.cv_requests_path_for_spec(cv_spec_path)
        output_root = (
            self.root
            / "generated"
            / "exp_a"
            / "cv_runs"
            / "nefg"
        )
        original_fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        original_score_rows = read_csv_manifest(output_root / "candidate_scores.csv")
        original_best_candidate = OmegaConf.load(output_root / "best_candidate.yaml")
        self.assertTrue(original_fold_rows)
        self.assertTrue(original_score_rows)

        (output_root / "fold_scores.csv").unlink()
        (output_root / "candidate_scores.csv").unlink()
        (output_root / "best_candidate.yaml").unlink()
        Path(manifest_path).unlink()

        refreshed_manifest_path = cv_runner.refresh_cv_scores_from_requests(requests_path)

        self.assertEqual(Path(refreshed_manifest_path), Path(manifest_path))
        refreshed_fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        refreshed_score_rows = read_csv_manifest(output_root / "candidate_scores.csv")
        refreshed_best_candidate = OmegaConf.load(output_root / "best_candidate.yaml")
        refreshed_manifest_rows = read_csv_manifest(refreshed_manifest_path)

        self.assertEqual(len(refreshed_fold_rows), len(original_fold_rows))
        self.assertEqual(len(refreshed_score_rows), len(original_score_rows))
        self.assertEqual(len(refreshed_manifest_rows), 1)
        self.assertEqual(
            str(refreshed_best_candidate.candidate_slug),
            str(original_best_candidate.candidate_slug),
        )

        original_fold_lookup = {
            (row["candidate_slug"], row["cv_fold_id"]): row for row in original_fold_rows
        }
        for row in refreshed_fold_rows:
            baseline = original_fold_lookup[(row["candidate_slug"], row["cv_fold_id"])]
            self.assertEqual(row["status"], baseline["status"])
            self.assertAlmostEqual(
                float(row["validation_loss"]),
                float(baseline["validation_loss"]),
                places=12,
            )
            self.assertAlmostEqual(
                float(row["validation_brier_score"]),
                float(baseline["validation_brier_score"]),
                places=12,
            )
            self.assertAlmostEqual(
                float(row["validation_ece"]),
                float(baseline["validation_ece"]),
                places=12,
            )
            self.assertAlmostEqual(
                float(row["validation_mean_magnetization_abs_diff"]),
                float(baseline["validation_mean_magnetization_abs_diff"]),
                places=12,
            )
            self.assertEqual(
                row["validation_observed_mean_magnetization"],
                baseline["validation_observed_mean_magnetization"],
            )
            self.assertEqual(
                row["validation_sampled_mean_magnetization_mean"],
                baseline["validation_sampled_mean_magnetization_mean"],
            )
            self.assertEqual(
                int(row["num_post_s_validation_slots"]),
                int(baseline["num_post_s_validation_slots"]),
            )
            self.assertEqual(
                row["post_s_validation_loss"],
                baseline["post_s_validation_loss"],
            )
            self.assertEqual(
                row["post_s_validation_brier_score"],
                baseline["post_s_validation_brier_score"],
            )
            self.assertEqual(
                row["post_s_validation_ece"],
                baseline["post_s_validation_ece"],
            )
            self.assertEqual(
                row["post_s_validation_mean_magnetization_abs_diff"],
                baseline["post_s_validation_mean_magnetization_abs_diff"],
            )
            self.assertEqual(
                row["post_s_validation_observed_mean_magnetization"],
                baseline["post_s_validation_observed_mean_magnetization"],
            )
            self.assertEqual(
                row["post_s_validation_sampled_mean_magnetization_mean"],
                baseline["post_s_validation_sampled_mean_magnetization_mean"],
            )

    def test_refresh_cv_scores_rebuilds_fold_scores_from_corrupted_intermediate_csv(
        self,
    ) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "rcs",
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0, 1]}},
                }
            ]
        )

        manifest_path = cv_runner.run_cv_folds(
            generation_manifest,
            cv_spec_path,
            execution_mode="validation",
            overwrite=True,
        )
        requests_path = cv_runner.validation_requests_path_for_spec(cv_spec_path)
        output_root = (
            self.root
            / "generated"
            / "exp_a"
            / "validation_runs"
            / "rcs"
        )
        original_fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        self.assertEqual(len(original_fold_rows), 2)

        write_csv_rows(output_root / "fold_scores.csv", [original_fold_rows[0]])
        refreshed_manifest_path = cv_runner.refresh_cv_scores_from_requests(
            requests_path,
            execution_mode="validation",
        )

        self.assertEqual(Path(refreshed_manifest_path), Path(manifest_path))
        refreshed_fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        self.assertEqual(len(refreshed_fold_rows), len(original_fold_rows))
        self.assertEqual(
            {
                (row["candidate_slug"], row["cv_fold_id"])
                for row in refreshed_fold_rows
            },
            {
                (row["candidate_slug"], row["cv_fold_id"])
                for row in original_fold_rows
            },
        )

    def test_validation_mode_runs_only_fold_one_and_writes_validation_manifest(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "nefg",
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0, 1]}},
                }
            ]
        )

        manifest_path = cv_runner.run_cv_folds(
            generation_manifest,
            cv_spec_path,
            execution_mode="validation",
            overwrite=True,
        )

        self.assertEqual(
            Path(manifest_path),
            self.root / "generated" / "validation_manifest.csv",
        )
        manifest_rows = read_csv_manifest(manifest_path)
        self.assertEqual(len(manifest_rows), 1)
        self.assertEqual(manifest_rows[0]["execution_mode"], "validation")

        requests_path = cv_runner.validation_requests_path_for_spec(cv_spec_path)
        request_rows = read_csv_manifest(requests_path)
        self.assertEqual(len(request_rows), 2)
        self.assertTrue(all(row["cv_fold_id"] == "1" for row in request_rows))

        output_root = (
            self.root
            / "generated"
            / "exp_a"
            / "validation_runs"
            / "nefg"
        )
        fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        self.assertEqual(len(fold_rows), 2)
        self.assertTrue(all(row["cv_fold_id"] == "1" for row in fold_rows))
        self.assertTrue((output_root / "best_candidate.yaml").exists())

    def test_validation_refresh_scores_rebuilds_validation_artifacts(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "nefg",
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0, 1]}},
                }
            ]
        )

        manifest_path = cv_runner.run_cv_folds(
            generation_manifest,
            cv_spec_path,
            execution_mode="validation",
            overwrite=True,
        )
        requests_path = cv_runner.validation_requests_path_for_spec(cv_spec_path)
        output_root = (
            self.root
            / "generated"
            / "exp_a"
            / "validation_runs"
            / "nefg"
        )
        original_fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        self.assertTrue(original_fold_rows)

        (output_root / "fold_scores.csv").unlink()
        (output_root / "candidate_scores.csv").unlink()
        (output_root / "best_candidate.yaml").unlink()
        Path(manifest_path).unlink()

        refreshed_manifest_path = cv_runner.refresh_cv_scores_from_requests(
            requests_path,
            execution_mode="validation",
        )

        self.assertEqual(Path(refreshed_manifest_path), Path(manifest_path))
        refreshed_fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        self.assertEqual(len(refreshed_fold_rows), len(original_fold_rows))
        self.assertTrue(all(row["execution_mode"] == "validation" for row in refreshed_fold_rows))

    def test_test_train_cv_split_runs_all_inner_folds_in_cv_mode(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 12, "N": 72}, "graph": {"params": {"p": 0.08}}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)

        class FakePyMetis:
            @staticmethod
            def part_graph(nparts, adjacency=None, recursive=None, contiguous=None):
                n = len(adjacency or [])
                if n <= 0:
                    return 0, []
                return 0, [min((index * int(nparts)) // n, int(nparts) - 1) for index in range(n)]

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=FakePyMetis(),
        ):
            split_output_paths = uscounty_splits.create_validation_test_splits(
                generation_manifest,
                outer_num_folds=3,
                inner_num_folds=3,
                overwrite=True,
            )
        split_output_root = split_output_paths[0]
        with np.load(io_path(split_output_root / "model_selection_folds.npz"), allow_pickle=False) as data:
            training_masks = np.asarray(data["training_masks"], dtype=bool)
            validation_masks = np.asarray(data["validation_masks"], dtype=bool)
        supported_fold_ids = {
            int(fold_index + 1)
            for fold_index in range(int(training_masks.shape[0]))
            if int(np.count_nonzero(training_masks[fold_index])) > 0
            and int(np.count_nonzero(validation_masks[fold_index])) > 0
        }

        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "split_backed_grid",
                    "split_kind": "test_train_cv",
                    "outer_num_folds": 3,
                    "test_fold_id": 1,
                    "num_folds": 3,
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0, 1]}},
                }
            ]
        )

        manifest_path = cv_runner.run_cv_folds(
            generation_manifest,
            cv_spec_path,
            overwrite=True,
        )

        self.assertEqual(Path(manifest_path), self.root / "generated" / "cv_manifest.csv")
        requests_path = cv_runner.cv_requests_path_for_spec(cv_spec_path)
        request_rows = read_csv_manifest(requests_path)
        self.assertEqual(len(request_rows), 2 * len(supported_fold_ids))
        self.assertEqual({row["split_kind"] for row in request_rows}, {"test_train_cv"})
        self.assertEqual({row["outer_num_folds"] for row in request_rows}, {"3"})
        self.assertEqual({row["test_fold_id"] for row in request_rows}, {"1"})
        self.assertEqual(
            {int(row["cv_fold_id"]) for row in request_rows},
            supported_fold_ids,
        )

        output_root = (
            self.root / "generated" / "exp_a" / "cv_runs" / "split_backed_grid"
        )
        fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        self.assertEqual(len(fold_rows), 2 * len(supported_fold_ids))
        self.assertEqual({row["split_kind"] for row in fold_rows}, {"test_train_cv"})
        self.assertEqual({row["outer_num_folds"] for row in fold_rows}, {"3"})
        self.assertEqual({row["test_fold_id"] for row in fold_rows}, {"1"})
        self.assertEqual({int(row["cv_fold_id"]) for row in fold_rows}, supported_fold_ids)
        self.assertEqual({row["status"] for row in fold_rows}, {"completed"})

    def test_test_train_cv_split_runs_only_fold_one_in_validation_mode(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 12, "N": 72}, "graph": {"params": {"p": 0.08}}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)

        class FakePyMetis:
            @staticmethod
            def part_graph(nparts, adjacency=None, recursive=None, contiguous=None):
                n = len(adjacency or [])
                if n <= 0:
                    return 0, []
                return 0, [min((index * int(nparts)) // n, int(nparts) - 1) for index in range(n)]

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=FakePyMetis(),
        ):
            split_output_paths = uscounty_splits.create_validation_test_splits(
                generation_manifest,
                outer_num_folds=3,
                inner_num_folds=3,
                overwrite=True,
            )
        split_output_root = split_output_paths[0]
        with np.load(io_path(split_output_root / "model_selection_folds.npz"), allow_pickle=False) as data:
            training_masks = np.asarray(data["training_masks"], dtype=bool)
            validation_masks = np.asarray(data["validation_masks"], dtype=bool)
        supported_fold_ids = [
            int(fold_index + 1)
            for fold_index in range(int(training_masks.shape[0]))
            if int(np.count_nonzero(training_masks[fold_index])) > 0
            and int(np.count_nonzero(validation_masks[fold_index])) > 0
        ]
        first_supported_fold_id = supported_fold_ids[0]

        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "split_backed_grid",
                    "split_kind": "test_train_cv",
                    "outer_num_folds": 3,
                    "test_fold_id": 1,
                    "num_folds": 3,
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0, 1]}},
                }
            ]
        )

        manifest_path = cv_runner.run_cv_folds(
            generation_manifest,
            cv_spec_path,
            execution_mode="validation",
            overwrite=True,
        )

        self.assertEqual(
            Path(manifest_path),
            self.root / "generated" / "validation_manifest.csv",
        )
        requests_path = cv_runner.validation_requests_path_for_spec(cv_spec_path)
        request_rows = read_csv_manifest(requests_path)
        self.assertEqual(len(request_rows), 2)
        self.assertTrue(all(int(row["cv_fold_id"]) == first_supported_fold_id for row in request_rows))
        self.assertTrue(all(row["split_kind"] == "test_train_cv" for row in request_rows))

        output_root = (
            self.root / "generated" / "exp_a" / "validation_runs" / "split_backed_grid"
        )
        fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        self.assertEqual(len(fold_rows), 2)
        self.assertTrue(all(int(row["cv_fold_id"]) == first_supported_fold_id for row in fold_rows))
        self.assertTrue(all(row["split_kind"] == "test_train_cv" for row in fold_rows))

    def test_collect_cv_manifest_from_requests_rebuilds_manifest_only(self) -> None:
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "nefg",
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0, 1]}},
                }
            ]
        )

        manifest_path = cv_runner.run_cv_folds(
            generation_manifest,
            cv_spec_path,
            overwrite=True,
        )
        requests_path = cv_runner.cv_requests_path_for_spec(cv_spec_path)
        output_root = (
            self.root
            / "generated"
            / "exp_a"
            / "cv_runs"
            / "nefg"
        )
        original_fold_rows = read_csv_manifest(output_root / "fold_scores.csv")
        original_score_rows = read_csv_manifest(output_root / "candidate_scores.csv")
        original_best_candidate = OmegaConf.load(output_root / "best_candidate.yaml")
        Path(manifest_path).unlink()
        (output_root / "candidate_scores.csv").unlink()
        (output_root / "best_candidate.yaml").unlink()

        with mock.patch.object(
            cv_runner,
            "evaluate_saved_fit_fold_metrics",
            side_effect=AssertionError("score refresh should not run"),
        ):
            collected_manifest_path = cv_runner.collect_cv_manifest_from_requests(
                requests_path
            )

        self.assertEqual(Path(collected_manifest_path), Path(manifest_path))
        self.assertEqual(
            read_csv_manifest(output_root / "fold_scores.csv"),
            original_fold_rows,
        )
        self.assertEqual(
            read_csv_manifest(output_root / "candidate_scores.csv"),
            original_score_rows,
        )
        collected_best_candidate = OmegaConf.load(output_root / "best_candidate.yaml")
        collected_manifest_rows = read_csv_manifest(collected_manifest_path)
        self.assertEqual(len(collected_manifest_rows), 1)
        self.assertEqual(
            collected_manifest_rows[0]["best_candidate_slug"],
            str(collected_best_candidate.candidate_slug),
        )
        self.assertEqual(
            str(collected_best_candidate.candidate_slug),
            str(original_best_candidate.candidate_slug),
        )

    def test_validation_brier_score_matches_spin_probability_formula(self) -> None:
        x = np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        h_x = np.asarray([[0.0, 0.2], [-0.7, 1.1]], dtype=float)
        mask = np.asarray([[True, False], [True, True]], dtype=bool)

        observed_positive = (x + 1.0) / 2.0
        predicted_positive = (1.0 + np.tanh(h_x)) / 2.0
        expected = float(
            np.mean(((observed_positive - predicted_positive) ** 2)[mask])
        )

        actual = validation_metrics.validation_brier_score(
            x=x,
            h_x=h_x,
            loss_mask=mask,
        )
        self.assertAlmostEqual(actual, expected, places=12)

    def test_validation_expected_calibration_error_matches_hand_computation(self) -> None:
        x = np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
        predicted_positive = np.asarray([[0.05, 0.15], [0.75, 0.95]], dtype=float)
        h_x = np.arctanh((2.0 * predicted_positive) - 1.0)
        mask = np.asarray([[True, True], [True, True]], dtype=bool)

        expected = (
            0.25 * abs(0.0 - 0.05)
            + 0.25 * abs(1.0 - 0.15)
            + 0.25 * abs(0.0 - 0.75)
            + 0.25 * abs(1.0 - 0.95)
        )

        actual = validation_metrics.validation_expected_calibration_error(
            x=x,
            h_x=h_x,
            loss_mask=mask,
        )
        self.assertAlmostEqual(actual, expected, places=12)

    def test_validation_statistics_ignore_masked_entries(self) -> None:
        x = np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        h_x = np.asarray([[0.4, -0.2], [0.9, -1.2]], dtype=float)
        mask = np.asarray([[False, True], [False, True]], dtype=bool)

        masked_brier = validation_metrics.validation_brier_score(
            x=x,
            h_x=h_x,
            loss_mask=mask,
        )
        masked_ece = validation_metrics.validation_expected_calibration_error(
            x=x,
            h_x=h_x,
            loss_mask=mask,
        )

        perturbed_x = np.asarray(x, dtype=float).copy()
        perturbed_h_x = np.asarray(h_x, dtype=float).copy()
        perturbed_x[0, 0] = -1.0
        perturbed_x[1, 0] = 1.0
        perturbed_h_x[0, 0] = 3.0
        perturbed_h_x[1, 0] = -3.0

        self.assertAlmostEqual(
            validation_metrics.validation_brier_score(
                x=perturbed_x,
                h_x=perturbed_h_x,
                loss_mask=mask,
            ),
            masked_brier,
            places=12,
        )
        self.assertAlmostEqual(
            validation_metrics.validation_expected_calibration_error(
                x=perturbed_x,
                h_x=perturbed_h_x,
                loss_mask=mask,
            ),
            masked_ece,
            places=12,
        )

    def test_evaluate_fold_metrics_conditions_on_separator_but_scores_validation_only(self) -> None:
        experiment_root = Path("unused-experiment-root")
        fit_root = Path("unused-fit-root")
        panel_context = {
            "x": np.asarray([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float),
            "z": np.zeros((3, 2), dtype=float),
            "x_0": np.asarray([1.0, -1.0], dtype=float),
            "s": 1,
            "e": 3,
        }
        bundle = SimpleNamespace(
            field_matrix=np.zeros((3, 2), dtype=float),
            beta=0.0,
            xi=1.0,
            eta=0.5,
            gamma_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            beta_mask_pre_s=False,
            beta_mask_post_e=False,
        )
        training_loss_mask = np.asarray(
            [[True, False], [False, False], [False, False]],
            dtype=bool,
        )
        validation_loss_mask = np.asarray(
            [[False, True], [False, False], [False, True]],
            dtype=bool,
        )

        with mock.patch.object(
            validation_metrics,
            "load_experiment_panel_context",
            return_value=panel_context,
        ), mock.patch.object(
            validation_metrics,
            "load_fit_parameter_bundle",
            return_value=bundle,
        ):
            metrics = validation_metrics.evaluate_saved_fit_fold_metrics(
                fit_root,
                experiment_root,
                training_loss_mask=training_loss_mask,
                validation_loss_mask=validation_loss_mask,
            )
            fit_loss = float(metrics["fit_loss"])
            validation_loss = float(metrics["validation_loss"])
            validation_brier = float(metrics["validation_brier_score"])
            validation_ece = float(metrics["validation_ece"])
            num_post_s_validation_slots = int(metrics["num_post_s_validation_slots"])
            post_s_validation_loss = metrics["post_s_validation_loss"]
            post_s_validation_brier = metrics["post_s_validation_brier_score"]
            post_s_validation_ece = metrics["post_s_validation_ece"]

        interaction_effect_x = interaction_effect(panel_context["x"], bundle.gamma_matrix)
        expected_fit_loss = evaluate_mple_loss_from_parts(
            x=panel_context["x"],
            z=panel_context["z"],
            x_0=panel_context["x_0"],
            field_matrix=bundle.field_matrix,
            beta=bundle.beta,
            xi=bundle.xi,
            eta=bundle.eta,
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params={},
            loss_mask=training_loss_mask,
            s=int(panel_context["s"]),
            e=int(panel_context["e"]),
            beta_mask_pre_s=bundle.beta_mask_pre_s,
            beta_mask_post_e=bundle.beta_mask_post_e,
        )
        expected_validation_loss = evaluate_mple_loss_from_parts(
            x=panel_context["x"],
            z=panel_context["z"],
            x_0=panel_context["x_0"],
            field_matrix=bundle.field_matrix,
            beta=bundle.beta,
            xi=bundle.xi,
            eta=bundle.eta,
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params={},
            loss_mask=validation_loss_mask,
            s=int(panel_context["s"]),
            e=int(panel_context["e"]),
            beta_mask_pre_s=bundle.beta_mask_pre_s,
            beta_mask_post_e=bundle.beta_mask_post_e,
        )
        post_s_validation_loss_mask = validation_loss_mask & validation_metrics.time_window_mask(
            t_steps=panel_context["x"].shape[0],
            n_nodes=panel_context["x"].shape[1],
            start_t=int(panel_context["s"]),
        )
        expected_post_s_validation_loss = evaluate_mple_loss_from_parts(
            x=panel_context["x"],
            z=panel_context["z"],
            x_0=panel_context["x_0"],
            field_matrix=bundle.field_matrix,
            beta=bundle.beta,
            xi=bundle.xi,
            eta=bundle.eta,
            interaction_effect_x=interaction_effect_x,
            fixed_scalar_params={},
            loss_mask=post_s_validation_loss_mask,
            s=int(panel_context["s"]),
            e=int(panel_context["e"]),
            beta_mask_pre_s=bundle.beta_mask_pre_s,
            beta_mask_post_e=bundle.beta_mask_post_e,
        )
        prev_x = np.vstack([panel_context["x_0"], panel_context["x"][:-1, :]])
        h_x = bundle.field_matrix + (bundle.beta * panel_context["z"]) + (bundle.xi * interaction_effect_x) + (bundle.eta * prev_x)
        expected_brier = validation_metrics.validation_brier_score(
            x=panel_context["x"],
            h_x=h_x,
            loss_mask=validation_loss_mask,
        )
        expected_ece = validation_metrics.validation_expected_calibration_error(
            x=panel_context["x"],
            h_x=h_x,
            loss_mask=validation_loss_mask,
        )
        expected_post_s_brier = validation_metrics.validation_brier_score(
            x=panel_context["x"],
            h_x=h_x,
            loss_mask=post_s_validation_loss_mask,
        )
        expected_post_s_ece = validation_metrics.validation_expected_calibration_error(
            x=panel_context["x"],
            h_x=h_x,
            loss_mask=post_s_validation_loss_mask,
        )

        self.assertAlmostEqual(fit_loss, float(expected_fit_loss), places=12)
        self.assertAlmostEqual(validation_loss, float(expected_validation_loss), places=12)
        self.assertAlmostEqual(validation_brier, expected_brier, places=12)
        self.assertAlmostEqual(validation_ece, expected_ece, places=12)
        self.assertEqual(num_post_s_validation_slots, 1)
        self.assertAlmostEqual(
            float(post_s_validation_loss),
            float(expected_post_s_validation_loss),
            places=12,
        )
        self.assertAlmostEqual(post_s_validation_brier, expected_post_s_brier, places=12)
        self.assertAlmostEqual(post_s_validation_ece, expected_post_s_ece, places=12)

        separator_flipped_context = dict(panel_context)
        separator_flipped_context["x"] = np.asarray(panel_context["x"], dtype=float).copy()
        separator_flipped_context["x"][1, 1] = -1.0
        with mock.patch.object(
            validation_metrics,
            "load_experiment_panel_context",
            return_value=separator_flipped_context,
        ), mock.patch.object(
            validation_metrics,
            "load_fit_parameter_bundle",
            return_value=bundle,
        ):
            flipped_metrics = validation_metrics.evaluate_saved_fit_fold_metrics(
                fit_root,
                experiment_root,
                training_loss_mask=training_loss_mask,
                validation_loss_mask=validation_loss_mask,
            )
            flipped_validation_loss = float(flipped_metrics["validation_loss"])
            flipped_validation_brier = float(flipped_metrics["validation_brier_score"])
            flipped_validation_ece = float(flipped_metrics["validation_ece"])
            flipped_post_s_validation_loss = flipped_metrics["post_s_validation_loss"]
            flipped_post_s_validation_brier = flipped_metrics[
                "post_s_validation_brier_score"
            ]
            flipped_post_s_validation_ece = flipped_metrics["post_s_validation_ece"]

        self.assertNotAlmostEqual(flipped_validation_loss, validation_loss, places=12)
        self.assertNotAlmostEqual(flipped_validation_brier, validation_brier, places=12)
        self.assertNotAlmostEqual(flipped_validation_ece, validation_ece, places=12)
        self.assertNotAlmostEqual(
            float(flipped_post_s_validation_loss),
            float(post_s_validation_loss),
            places=12,
        )
        self.assertNotAlmostEqual(
            flipped_post_s_validation_brier,
            post_s_validation_brier,
            places=12,
        )
        self.assertNotAlmostEqual(
            flipped_post_s_validation_ece,
            post_s_validation_ece,
            places=12,
        )

    def test_evaluate_fold_metrics_leaves_post_s_metrics_blank_when_no_post_s_validation_support(
        self,
    ) -> None:
        experiment_root = Path("unused-experiment-root")
        fit_root = Path("unused-fit-root")
        panel_context = {
            "x": np.asarray([[1.0, -1.0], [1.0, 1.0]], dtype=float),
            "z": np.zeros((2, 2), dtype=float),
            "x_0": np.asarray([1.0, -1.0], dtype=float),
            "s": 1,
            "e": 2,
        }
        bundle = SimpleNamespace(
            field_matrix=np.zeros((2, 2), dtype=float),
            beta=0.0,
            xi=1.0,
            eta=0.0,
            gamma_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            beta_mask_pre_s=False,
            beta_mask_post_e=False,
        )
        training_loss_mask = np.asarray([[False, False], [True, False]], dtype=bool)
        validation_loss_mask = np.asarray([[False, True], [False, False]], dtype=bool)

        with mock.patch.object(
            validation_metrics,
            "load_experiment_panel_context",
            return_value=panel_context,
        ), mock.patch.object(
            validation_metrics,
            "load_fit_parameter_bundle",
            return_value=bundle,
        ):
            metrics = validation_metrics.evaluate_saved_fit_fold_metrics(
                fit_root,
                experiment_root,
                training_loss_mask=training_loss_mask,
                validation_loss_mask=validation_loss_mask,
            )
            validation_loss = float(metrics["validation_loss"])
            validation_brier = float(metrics["validation_brier_score"])
            validation_ece = float(metrics["validation_ece"])
            num_post_s_validation_slots = int(metrics["num_post_s_validation_slots"])
            post_s_validation_loss = metrics["post_s_validation_loss"]
            post_s_validation_brier = metrics["post_s_validation_brier_score"]
            post_s_validation_ece = metrics["post_s_validation_ece"]

        self.assertGreater(float(validation_loss), 0.0)
        self.assertGreater(float(validation_brier), 0.0)
        self.assertGreater(float(validation_ece), 0.0)
        self.assertEqual(num_post_s_validation_slots, 0)
        self.assertIsNone(post_s_validation_loss)
        self.assertIsNone(post_s_validation_brier)
        self.assertIsNone(post_s_validation_ece)

    def test_evaluate_test_metrics_renames_fold_metrics_consistently(self) -> None:
        panel_context = {
            "x": np.asarray([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float),
            "z": np.zeros((3, 2), dtype=float),
            "x_0": np.asarray([1.0, -1.0], dtype=float),
            "s": 1,
            "e": 3,
        }
        bundle = SimpleNamespace(
            field_matrix=np.zeros((3, 2), dtype=float),
            beta=0.0,
            xi=1.0,
            eta=0.5,
            gamma_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            beta_mask_pre_s=False,
            beta_mask_post_e=False,
        )
        training_loss_mask = np.asarray(
            [[True, False], [False, False], [False, False]],
            dtype=bool,
        )
        test_loss_mask = np.asarray(
            [[False, True], [False, False], [False, True]],
            dtype=bool,
        )

        baseline = validation_metrics.evaluate_fold_metrics(
            panel_context=panel_context,
            bundle=bundle,
            training_loss_mask=training_loss_mask,
            validation_loss_mask=test_loss_mask,
            validation_sampling={"num_samples": 2, "gibbs_sweeps": 1, "seed": 0},
        )
        test_metrics = validation_metrics.evaluate_test_metrics(
            panel_context=panel_context,
            bundle=bundle,
            training_loss_mask=training_loss_mask,
            test_loss_mask=test_loss_mask,
            sampling={"num_samples": 2, "gibbs_sweeps": 1, "seed": 0},
        )

        self.assertAlmostEqual(test_metrics["training_loss"], baseline["fit_loss"], places=12)
        self.assertEqual(test_metrics["num_training_slots"], 1)
        self.assertAlmostEqual(test_metrics["test_loss"], baseline["validation_loss"], places=12)
        self.assertEqual(test_metrics["num_test_slots"], 2)
        self.assertAlmostEqual(
            test_metrics["test_brier_score"],
            baseline["validation_brier_score"],
            places=12,
        )
        self.assertAlmostEqual(test_metrics["test_ece"], baseline["validation_ece"], places=12)
        self.assertEqual(
            test_metrics["num_post_s_test_slots"],
            baseline["num_post_s_validation_slots"],
        )
        self.assertEqual(test_metrics["post_s_test_loss"], baseline["post_s_validation_loss"])
        self.assertEqual(
            test_metrics["post_s_test_brier_score"],
            baseline["post_s_validation_brier_score"],
        )
        self.assertEqual(test_metrics["post_s_test_ece"], baseline["post_s_validation_ece"])
        self.assertEqual(
            test_metrics["test_mean_magnetization_abs_diff"],
            baseline["validation_mean_magnetization_abs_diff"],
        )
        self.assertEqual(
            test_metrics["test_observed_mean_magnetization"],
            baseline["validation_observed_mean_magnetization"],
        )
        self.assertEqual(
            test_metrics["test_sampled_mean_magnetization_mean"],
            baseline["validation_sampled_mean_magnetization_mean"],
        )

    def test_evaluate_test_metrics_leaves_post_s_test_metrics_blank_when_no_post_s_support(
        self,
    ) -> None:
        panel_context = {
            "x": np.asarray([[1.0, -1.0], [1.0, 1.0]], dtype=float),
            "z": np.zeros((2, 2), dtype=float),
            "x_0": np.asarray([1.0, -1.0], dtype=float),
            "s": 1,
            "e": 2,
        }
        bundle = SimpleNamespace(
            field_matrix=np.zeros((2, 2), dtype=float),
            beta=0.0,
            xi=1.0,
            eta=0.0,
            gamma_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            beta_mask_pre_s=False,
            beta_mask_post_e=False,
        )
        training_loss_mask = np.asarray([[False, False], [True, False]], dtype=bool)
        test_loss_mask = np.asarray([[False, True], [False, False]], dtype=bool)

        metrics = validation_metrics.evaluate_test_metrics(
            panel_context=panel_context,
            bundle=bundle,
            training_loss_mask=training_loss_mask,
            test_loss_mask=test_loss_mask,
        )
        self.assertEqual(metrics["num_post_s_test_slots"], 0)
        self.assertIsNone(metrics["post_s_test_loss"])
        self.assertIsNone(metrics["post_s_test_brier_score"])
        self.assertIsNone(metrics["post_s_test_ece"])

    def test_candidate_score_sort_key_prefers_lower_post_s_mag_diff_then_brier_then_loss(
        self,
    ) -> None:
        rows = [
            {
                "candidate_slug": "higher_loss_same_mag_same_brier",
                "candidate_index": 2,
                "weighted_mean_validation_mean_magnetization_abs_diff": 0.01,
                "weighted_mean_validation_brier_score": 0.01,
                "weighted_mean_validation_loss": 0.01,
                "weighted_mean_post_s_validation_mean_magnetization_abs_diff": 0.10,
                "weighted_mean_post_s_validation_brier_score": 0.10,
                "weighted_mean_post_s_validation_loss": 0.60,
            },
            {
                "candidate_slug": "lower_loss_same_mag_same_brier",
                "candidate_index": 3,
                "weighted_mean_validation_mean_magnetization_abs_diff": 0.01,
                "weighted_mean_validation_brier_score": 0.01,
                "weighted_mean_validation_loss": 0.01,
                "weighted_mean_post_s_validation_mean_magnetization_abs_diff": 0.10,
                "weighted_mean_post_s_validation_brier_score": 0.10,
                "weighted_mean_post_s_validation_loss": 0.40,
            },
            {
                "candidate_slug": "worse_mag",
                "candidate_index": 1,
                "weighted_mean_validation_mean_magnetization_abs_diff": 0.01,
                "weighted_mean_validation_brier_score": 0.01,
                "weighted_mean_validation_loss": 0.01,
                "weighted_mean_post_s_validation_mean_magnetization_abs_diff": 0.12,
                "weighted_mean_post_s_validation_brier_score": 0.05,
                "weighted_mean_post_s_validation_loss": 0.20,
            },
            {
                "candidate_slug": "same_mag_same_brier_same_loss_lower_index",
                "candidate_index": 1,
                "weighted_mean_validation_mean_magnetization_abs_diff": 0.01,
                "weighted_mean_validation_brier_score": 0.01,
                "weighted_mean_validation_loss": 0.01,
                "weighted_mean_post_s_validation_mean_magnetization_abs_diff": 0.10,
                "weighted_mean_post_s_validation_brier_score": 0.10,
                "weighted_mean_post_s_validation_loss": 0.40,
            },
            {
                "candidate_slug": "better_brier_same_mag",
                "candidate_index": 4,
                "weighted_mean_validation_mean_magnetization_abs_diff": 0.99,
                "weighted_mean_validation_brier_score": 0.99,
                "weighted_mean_validation_loss": 0.99,
                "weighted_mean_post_s_validation_mean_magnetization_abs_diff": 0.10,
                "weighted_mean_post_s_validation_brier_score": 0.09,
                "weighted_mean_post_s_validation_loss": 0.90,
            },
        ]

        ordered = sorted(rows, key=validation_metrics.candidate_score_sort_key)

        self.assertEqual(
            [row["candidate_slug"] for row in ordered],
            [
                "better_brier_same_mag",
                "same_mag_same_brier_same_loss_lower_index",
                "lower_loss_same_mag_same_brier",
                "higher_loss_same_mag_same_brier",
                "worse_mag",
            ],
        )

    def test_standard_error_winner_rule_uses_post_s_metrics(
        self,
    ) -> None:
        candidates = [
            {
                "name": "best_mag_more_regularized",
                "slug": "best_mag_more_regularized",
                "_candidate_index": 1,
                "lambda_nuclear": 0.0,
                "lambda_frobenius": 0.0,
                "lambda_uv_ridge": 0.5,
                "v_column_l2_max": None,
            },
            {
                "name": "within_se_less_regularized",
                "slug": "within_se_less_regularized",
                "_candidate_index": 2,
                "lambda_nuclear": 0.0,
                "lambda_frobenius": 0.0,
                "lambda_uv_ridge": 0.1,
                "v_column_l2_max": None,
            },
            {
                "name": "within_mag_se_but_bad_brier",
                "slug": "within_mag_se_but_bad_brier",
                "_candidate_index": 3,
                "lambda_nuclear": 0.0,
                "lambda_frobenius": 0.0,
                "lambda_uv_ridge": 0.0,
                "v_column_l2_max": None,
            },
            {
                "name": "outside_mag_se",
                "slug": "outside_mag_se",
                "_candidate_index": 4,
                "lambda_nuclear": 0.0,
                "lambda_frobenius": 0.0,
                "lambda_uv_ridge": 0.0,
                "v_column_l2_max": None,
            },
        ]
        candidate_score_rows = [
            {
                "candidate_slug": "best_mag_more_regularized",
                "status": "completed",
                "mean_fold_validation_mean_magnetization_abs_diff": 0.01,
                "standard_error_fold_validation_mean_magnetization_abs_diff": 0.001,
                "mean_fold_validation_brier_score": 0.01,
                "standard_error_fold_validation_brier_score": 0.001,
                "mean_fold_post_s_validation_mean_magnetization_abs_diff": 0.10,
                "standard_error_fold_post_s_validation_mean_magnetization_abs_diff": 0.02,
                "mean_fold_post_s_validation_brier_score": 0.20,
                "standard_error_fold_post_s_validation_brier_score": 0.01,
            },
            {
                "candidate_slug": "within_se_less_regularized",
                "status": "completed",
                "mean_fold_validation_mean_magnetization_abs_diff": 0.50,
                "standard_error_fold_validation_mean_magnetization_abs_diff": 0.001,
                "mean_fold_validation_brier_score": 0.50,
                "standard_error_fold_validation_brier_score": 0.001,
                "mean_fold_post_s_validation_mean_magnetization_abs_diff": 0.11,
                "standard_error_fold_post_s_validation_mean_magnetization_abs_diff": 0.01,
                "mean_fold_post_s_validation_brier_score": 0.205,
                "standard_error_fold_post_s_validation_brier_score": 0.02,
            },
            {
                "candidate_slug": "within_mag_se_but_bad_brier",
                "status": "completed",
                "mean_fold_validation_mean_magnetization_abs_diff": 0.40,
                "standard_error_fold_validation_mean_magnetization_abs_diff": 0.001,
                "mean_fold_validation_brier_score": 0.40,
                "standard_error_fold_validation_brier_score": 0.001,
                "mean_fold_post_s_validation_mean_magnetization_abs_diff": 0.11,
                "standard_error_fold_post_s_validation_mean_magnetization_abs_diff": 0.01,
                "mean_fold_post_s_validation_brier_score": 0.23,
                "standard_error_fold_post_s_validation_brier_score": 0.01,
            },
            {
                "candidate_slug": "outside_mag_se",
                "status": "completed",
                "mean_fold_validation_mean_magnetization_abs_diff": 0.02,
                "standard_error_fold_validation_mean_magnetization_abs_diff": 0.001,
                "mean_fold_validation_brier_score": 0.02,
                "standard_error_fold_validation_brier_score": 0.001,
                "mean_fold_post_s_validation_mean_magnetization_abs_diff": 0.14,
                "standard_error_fold_post_s_validation_mean_magnetization_abs_diff": 0.01,
                "mean_fold_post_s_validation_brier_score": 0.18,
                "standard_error_fold_post_s_validation_brier_score": 0.01,
            },
        ]

        best_candidate, best_row = cv_runner._select_best_candidate_within_standard_error(
            candidates,
            candidate_score_rows,
        )

        self.assertEqual(best_candidate["slug"], "within_se_less_regularized")
        self.assertEqual(best_row["candidate_slug"], "within_se_less_regularized")

    def test_standard_error_winner_rule_falls_back_when_post_s_metrics_missing(
        self,
    ) -> None:
        candidates = [
            {
                "name": "candidate_a",
                "slug": "candidate_a",
                "_candidate_index": 1,
                "lambda_nuclear": 0.0,
                "lambda_frobenius": 0.0,
                "lambda_uv_ridge": 0.5,
                "v_column_l2_max": None,
            },
            {
                "name": "candidate_b",
                "slug": "candidate_b",
                "_candidate_index": 2,
                "lambda_nuclear": 0.0,
                "lambda_frobenius": 0.0,
                "lambda_uv_ridge": 0.1,
                "v_column_l2_max": None,
            },
        ]
        candidate_score_rows = [
            {
                "candidate_slug": "candidate_a",
                "status": "completed",
                "mean_fold_validation_mean_magnetization_abs_diff": 0.10,
                "standard_error_fold_validation_mean_magnetization_abs_diff": 0.02,
                "mean_fold_validation_brier_score": 0.20,
                "standard_error_fold_validation_brier_score": 0.01,
                "mean_fold_post_s_validation_mean_magnetization_abs_diff": "",
                "standard_error_fold_post_s_validation_mean_magnetization_abs_diff": "",
                "mean_fold_post_s_validation_brier_score": "",
                "standard_error_fold_post_s_validation_brier_score": "",
            },
            {
                "candidate_slug": "candidate_b",
                "status": "completed",
                "mean_fold_validation_mean_magnetization_abs_diff": 0.11,
                "standard_error_fold_validation_mean_magnetization_abs_diff": 0.01,
                "mean_fold_validation_brier_score": 0.205,
                "standard_error_fold_validation_brier_score": 0.02,
                "mean_fold_post_s_validation_mean_magnetization_abs_diff": "",
                "standard_error_fold_post_s_validation_mean_magnetization_abs_diff": "",
                "mean_fold_post_s_validation_brier_score": "",
                "standard_error_fold_post_s_validation_brier_score": "",
            },
        ]

        best_candidate, best_row = cv_runner._select_best_candidate_within_standard_error(
            candidates,
            candidate_score_rows,
        )

        self.assertEqual(best_candidate["slug"], "candidate_b")
        self.assertEqual(best_row["candidate_slug"], "candidate_b")

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

    @unittest.skipIf(shutil.which("bash") is None, "bash is required for shell submission test")
    def test_submit_fit_jobs_outer_masked_without_search_slug_submits_all_searches(self) -> None:
        bash_path = shutil.which("bash")
        if bash_path is None or "system32" in bash_path.lower():
            self.skipTest("portable bash is not available in this environment")
        generation_spec_path = self._write_generation_spec([{"name": "exp_a", "dimensions": {"T": 9}}])
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest, overwrite=True)
        cv_spec_path = self._write_cv_spec(
            [
                {"name": "mask_grid", "grid": {"estimation": {"beta_mask_pre_s": [False]}}},
                {"name": "mask_grid_alt", "grid": {"estimation": {"beta_mask_pre_s": [True]}}},
            ]
        )
        cv_runner.run_cv_folds(generation_manifest, cv_spec_path, overwrite=True)
        fake_sbatch_path, fake_counter_path, fake_log_path = self._write_fake_sbatch()

        result = subprocess.run(
            [bash_path, "submit_fit_jobs.sh"],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "GENERATION_MANIFEST_PATH": str(generation_manifest),
                "CV_SPEC_PATH": str(cv_spec_path),
                "FIT_MODE": "outer_masked",
                "SBATCH_BIN": str(fake_sbatch_path),
                "WORKER_SCRIPT": "run_fit_job.sh",
                "FAKE_SBATCH_COUNTER": str(fake_counter_path),
                "FAKE_SBATCH_LOG": str(fake_log_path),
            },
        )

        log_lines = fake_log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log_lines), 3)
        self.assertIn("run_fit_job.sh", log_lines[0])
        self.assertIn("<exp_a>", log_lines[0])
        self.assertIn("<mask_grid>", log_lines[0])
        self.assertIn("run_fit_job.sh", log_lines[1])
        self.assertIn("<exp_a>", log_lines[1])
        self.assertIn("<mask_grid_alt>", log_lines[1])
        self.assertIn("--refresh_manifest", log_lines[2])
        self.assertNotIn("--search_slug", log_lines[2])
        self.assertEqual(result.stdout.strip(), "job3")

    @unittest.skipIf(shutil.which("bash") is None, "bash is required for shell submission test")
    def test_submit_cv_jobs_submits_workers_and_refresh_manifest(self) -> None:
        bash_path = shutil.which("bash")
        if bash_path is None or "system32" in bash_path.lower():
            self.skipTest("portable bash is not available in this environment")
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "nefg",
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0, 1]}},
                }
            ]
        )
        fake_sbatch_path, fake_counter_path, fake_log_path = self._write_fake_sbatch()

        result = subprocess.run(
            [bash_path, "submit_cv_jobs.sh"],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "GENERATION_MANIFEST_PATH": str(generation_manifest),
                "CV_SPEC_PATH": str(cv_spec_path),
                "EXECUTION_MODE": "validation",
                "SBATCH_BIN": str(fake_sbatch_path),
                "WORKER_SCRIPT": "run_cv_job.sh",
                "FAKE_SBATCH_COUNTER": str(fake_counter_path),
                "FAKE_SBATCH_LOG": str(fake_log_path),
            },
        )

        log_lines = fake_log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log_lines), 2)
        self.assertIn("run_cv_job.sh", log_lines[0])
        self.assertIn("<exp_a>", log_lines[0])
        self.assertIn("<nefg>", log_lines[0])
        self.assertIn("--refresh_manifest", log_lines[1])
        self.assertIn("--execution_mode 'validation'", log_lines[1])
        self.assertEqual(result.stdout.strip(), "job2")

    @unittest.skipIf(shutil.which("bash") is None, "bash is required for shell worker test")
    def test_run_cv_job_forwards_continue_flag(self) -> None:
        bash_path = shutil.which("bash")
        if bash_path is None or "system32" in bash_path.lower():
            self.skipTest("portable bash is not available in this environment")
        _, fake_pixi_log_path = self._write_fake_pixi()

        subprocess.run(
            [bash_path, "run_cv_job.sh", "exp_a", "search_a"],
            check=True,
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "PATH": f"{self.root}{os.pathsep}{os.environ.get('PATH', '')}",
                "FAKE_PIXI_LOG": str(fake_pixi_log_path),
                "CV_CONTINUE": "true",
                "SLURM_JOB_ID": "123",
                "SLURM_JOB_NAME": "cv",
            },
        )

        log_lines = fake_pixi_log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log_lines), 1)
        self.assertIn("run_cv_folds.py", log_lines[0])
        self.assertIn("--continue", log_lines[0])
        self.assertIn("--run_request", log_lines[0])

    @unittest.skipIf(shutil.which("bash") is None, "bash is required for shell submission test")
    def test_submit_cv_jobs_propagates_continue_flag(self) -> None:
        bash_path = shutil.which("bash")
        if bash_path is None or "system32" in bash_path.lower():
            self.skipTest("portable bash is not available in this environment")
        generation_spec_path = self._write_generation_spec(
            [{"name": "exp_a", "dimensions": {"T": 9}}]
        )
        generation_manifest = run_generation(generation_spec_path, overwrite=True)
        cv_folds.run_build_cv_folds(generation_manifest)
        cv_spec_path = self._write_cv_spec(
            [
                {
                    "name": "cmg",
                    "optimizer_mode": "no_external_field",
                    "grid": {"optimizer": {"seed": [0]}},
                }
            ]
        )
        fake_sbatch_path, fake_counter_path, fake_log_path = self._write_env_logging_sbatch()

        result = subprocess.run(
            [bash_path, "submit_cv_jobs.sh"],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "GENERATION_MANIFEST_PATH": str(generation_manifest),
                "CV_SPEC_PATH": str(cv_spec_path),
                "CV_CONTINUE": "true",
                "EXECUTION_MODE": "validation",
                "SBATCH_BIN": str(fake_sbatch_path),
                "WORKER_SCRIPT": "run_cv_job.sh",
                "FAKE_SBATCH_COUNTER": str(fake_counter_path),
                "FAKE_SBATCH_LOG": str(fake_log_path),
            },
        )

        log_lines = fake_log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log_lines), 2)
        self.assertIn("env:CV_CONTINUE=true", log_lines[0])
        self.assertIn("run_cv_job.sh", log_lines[0])
        self.assertIn("--refresh_manifest", log_lines[1])
        self.assertEqual(result.stdout.strip(), "job2")

    @unittest.skipIf(shutil.which("bash") is None, "bash is required for shell orchestration test")
    def test_run_tests_sh_waits_between_stage_submissions(self) -> None:
        bash_path = shutil.which("bash")
        if bash_path is None or "system32" in bash_path.lower():
            self.skipTest("portable bash is not available in this environment")
        log_path = self.root / "orchestration.log"
        generation_submitter = self.root / "fake_generation_submitter.sh"
        cv_submitter = self.root / "fake_cv_submitter.sh"
        fit_submitter = self.root / "fake_fit_submitter.sh"
        posterior_submitter = self.root / "fake_posterior_submitter.sh"
        build_cv_script = self.root / "fake_build_cv.sh"
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
        cv_submitter.write_text(
            "#!/bin/bash\nset -euo pipefail\nprintf 'model_selection_submit:%s\\n' \"${EXECUTION_MODE}\" >> \"${STAGE_LOG}\"\nprintf 'job-model-%s\\n' \"${EXECUTION_MODE}\"\n",
            encoding="utf-8",
        )
        posterior_submitter.write_text(
            "#!/bin/bash\nset -euo pipefail\nprintf 'posterior_submit\\n' >> \"${STAGE_LOG}\"\nprintf 'job-posterior\\n'\n",
            encoding="utf-8",
        )
        build_cv_script.write_text(
            "#!/bin/bash\nset -euo pipefail\nprintf 'build_cv\\n' >> \"${STAGE_LOG}\"\n",
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
            cv_submitter,
            fit_submitter,
            posterior_submitter,
            build_cv_script,
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
                "CV_SUBMITTER": str(cv_submitter),
                "FIT_SUBMITTER": str(fit_submitter),
                "POSTERIOR_PREDICTIVE_SUBMITTER": str(posterior_submitter),
                "BUILD_CV_FOLDS_SCRIPT": str(build_cv_script),
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
                "build_cv",
                "model_selection_submit:cv",
                "wait:job-model-cv",
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
        node_geography = pd.DataFrame(
            {
                "fips": node_order,
                "county": ["a", "b", "c", "d"],
                "state_name": ["Alabama"] * 4,
                "STATEFP": ["01"] * 4,
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
            },
        )
        return {
            "node_geography": node_geography,
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
            return_value=(pd.DataFrame(), fixture["node_geography"], fixture["centroids"]),
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

        self.assertEqual(dims, {"N": 4, "T": 4, "s": 1, "e": 4})
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
        self.assertEqual(derived_dims, {"N": 4, "T": 2, "s": 0, "e": 2})

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
        self.assertTrue(
            Path(
                io_path(counterfactual_root / "counterfactual_time_summary.csv")
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
        time_means: list[float],
        time_q025: list[float],
        time_q975: list[float],
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

        time_rows = [
            {
                "time_index": time_index,
                "sample_mean": mean,
                "sample_std": 0.0,
                "q025": q025,
                "q500": mean,
                "q975": q975,
                "num_finite_samples": 4,
            }
            for time_index, (mean, q025, q975) in enumerate(
                zip(time_means, time_q025, time_q975)
            )
        ]
        with (output_root / "counterfactual_time_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "time_index",
                    "sample_mean",
                    "sample_std",
                    "q025",
                    "q500",
                    "q975",
                    "num_finite_samples",
                ],
            )
            writer.writeheader()
            writer.writerows(time_rows)

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

    def test_compute_counterfactual_sample_summary_includes_time_mean_magnetization(
        self,
    ) -> None:
        x = np.asarray(
            [
                [1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )

        summary = compute_counterfactual_sample_summary(x, s=1)

        self.assertAlmostEqual(summary["overall_mean_magnetization"], float(np.mean(x)))
        self.assertAlmostEqual(
            summary["post_intervention_mean_magnetization"],
            float(np.mean(x[1:, :])),
        )
        self.assertTrue(np.allclose(summary["unit_mean_magnetization"], np.mean(x, axis=0)))
        self.assertTrue(np.allclose(summary["time_mean_magnetization"], np.mean(x, axis=1)))

    def test_summarize_observed_mean_statistics_reports_aggregate_errors(self) -> None:
        observed_summary = compute_counterfactual_sample_summary(
            np.asarray(
                [
                    [1.0, -1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                ],
                dtype=float,
            ),
            s=1,
        )
        sample_summaries = {
            "overall_mean_magnetization": np.asarray([0.0, 0.5, 0.25], dtype=float),
            "post_intervention_mean_magnetization": np.asarray(
                [0.5, 0.75, 0.25], dtype=float
            ),
            "unit_mean_magnetization": np.asarray(
                [
                    [0.00, 0.50],
                    [0.25, 0.75],
                    [0.50, 0.25],
                ],
                dtype=float,
            ),
            "time_mean_magnetization": np.asarray(
                [
                    [0.0, 1.0 / 3.0, 0.0],
                    [0.5, 1.0, 0.0],
                    [0.0, 2.0 / 3.0, 0.5],
                ],
                dtype=float,
            ),
        }

        mean_rows, unit_rows, time_rows, summary = summarize_observed_mean_statistics(
            observed_summary,
            sample_summaries,
        )

        self.assertEqual(len(mean_rows), 2)
        self.assertEqual(len(unit_rows), 2)
        self.assertEqual(len(time_rows), 3)
        self.assertAlmostEqual(
            float(summary["overall_mean_abs_error"]),
            1.0 / 12.0,
        )
        self.assertAlmostEqual(
            float(summary["post_intervention_mean_abs_error"]),
            0.0,
        )
        self.assertAlmostEqual(float(summary["unit_mean_abs_error_mean"]), 1.0 / 8.0)
        self.assertAlmostEqual(
            float(summary["unit_mean_rmse"]),
            float(np.sqrt(5.0 / 288.0)),
        )
        self.assertAlmostEqual(float(summary["unit_mean_max_abs_error"]), 1.0 / 6.0)
        self.assertAlmostEqual(
            float(summary["unit_mean_95_interval_coverage_rate"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(summary["time_mean_abs_error_mean"]),
            2.0 / 9.0,
        )
        self.assertAlmostEqual(
            float(summary["time_mean_rmse"]),
            float(np.sqrt(1.0 / 18.0)),
        )
        self.assertAlmostEqual(float(summary["time_mean_max_abs_error"]), 1.0 / 3.0)
        self.assertAlmostEqual(
            float(summary["time_mean_95_interval_coverage_rate"]),
            2.0 / 3.0,
        )

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

    def test_load_truth_parameter_bundle_returns_expected_truth_values(self) -> None:
        experiment_root = self.root / "exp_truth_bundle"
        experiment_root.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            OmegaConf.create(
                {
                    "estimation_params": {
                        "beta": 0.25,
                        "xi": -0.5,
                        "eta": 1.25,
                    }
                }
            ),
            experiment_root / "generation_realized_config.yaml",
        )
        save_model_artifacts(
            experiment_root,
            ModelArtifacts(
                gamma_matrix=np.eye(3, dtype=float),
                t_steps=4,
                latent_rank=2,
                field_matrix=np.full((4, 3), 0.75, dtype=float),
            ),
        )

        bundle = load_truth_parameter_bundle(experiment_root)

        self.assertEqual(bundle.source_type, "truth")
        self.assertEqual(bundle.source_name, "truth")
        self.assertAlmostEqual(bundle.beta, 0.25)
        self.assertAlmostEqual(bundle.xi, -0.5)
        self.assertAlmostEqual(bundle.eta, 1.25)
        self.assertEqual(bundle.latent_rank, 2)
        self.assertEqual(bundle.t_steps, 4)
        self.assertTrue(np.allclose(bundle.field_matrix, 0.75))
        self.assertTrue(np.array_equal(bundle.gamma_matrix, np.eye(3, dtype=float)))

    def test_load_fit_parameter_bundle_prefers_saved_bundle_npz(self) -> None:
        experiment_root = self.root / "exp_fit_bundle_saved"
        fit_root = experiment_root / "fits" / "rank_0"
        fit_root.mkdir(parents=True, exist_ok=True)
        save_model_artifacts(
            experiment_root,
            ModelArtifacts(
                gamma_matrix=np.zeros((2, 2), dtype=float),
                t_steps=3,
                latent_rank=1,
                field_matrix=np.full((3, 2), 9.0, dtype=float),
            ),
        )
        save_estimated_parameter_bundle(
            fit_root / "estimated_parameter_bundle.npz",
            beta=0.5,
            xi=0.1,
            eta=-0.2,
            latent_rank=1,
            t_steps=3,
            field_matrix=np.full((3, 2), 2.0, dtype=float),
        )
        with (fit_root / "mple_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "estimate"])
            writer.writeheader()
            writer.writerows(
                [
                    {"name": "beta", "estimate": "7.5"},
                    {"name": "xi", "estimate": "8.5"},
                    {"name": "eta", "estimate": "9.5"},
                ]
            )
        np.savez(
            fit_root / "estimated_field_artifacts.npz",
            field_matrix=np.full((3, 2), -4.0, dtype=float),
            latent_rank=np.asarray(7, dtype=int),
            t_steps=np.asarray(11, dtype=int),
        )

        bundle = load_fit_parameter_bundle(fit_root, experiment_root)

        self.assertAlmostEqual(bundle.beta, 0.5)
        self.assertAlmostEqual(bundle.xi, 0.1)
        self.assertAlmostEqual(bundle.eta, -0.2)
        self.assertEqual(bundle.latent_rank, 1)
        self.assertEqual(bundle.t_steps, 3)
        self.assertTrue(np.allclose(bundle.field_matrix, 2.0))

    def test_load_fit_parameter_bundle_fallback_reconstructs_values(self) -> None:
        experiment_root = self.root / "exp_fit_bundle_fallback"
        fit_root = experiment_root / "fits" / "rank_0"
        fit_root.mkdir(parents=True, exist_ok=True)
        save_model_artifacts(
            experiment_root,
            ModelArtifacts(
                gamma_matrix=np.eye(2, dtype=float),
                t_steps=5,
                latent_rank=0,
                field_matrix=np.zeros((5, 2), dtype=float),
            ),
        )
        with (fit_root / "mple_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "estimate"])
            writer.writeheader()
            writer.writerows(
                [
                    {"name": "beta", "estimate": "1.25"},
                    {"name": "xi", "estimate": "-0.75"},
                    {"name": "eta", "estimate": "0.5"},
                ]
            )
        np.savez(
            fit_root / "estimated_field_artifacts.npz",
            field_matrix=np.full((5, 2), 3.0, dtype=float),
            latent_rank=np.asarray(4, dtype=int),
            t_steps=np.asarray(5, dtype=int),
        )

        bundle = load_fit_parameter_bundle(fit_root, experiment_root)

        self.assertEqual(bundle.source_type, "fit")
        self.assertEqual(bundle.source_name, "rank_0")
        self.assertAlmostEqual(bundle.beta, 1.25)
        self.assertAlmostEqual(bundle.xi, -0.75)
        self.assertAlmostEqual(bundle.eta, 0.5)
        self.assertEqual(bundle.latent_rank, 4)
        self.assertEqual(bundle.t_steps, 5)
        self.assertTrue(np.allclose(bundle.field_matrix, 3.0))
        self.assertTrue(np.array_equal(bundle.gamma_matrix, np.eye(2, dtype=float)))

    def test_build_manifest_row_matches_metadata_reconstruction_for_observed(
        self,
    ) -> None:
        experiment_root = self.root / "manifest_observed"
        output_root = experiment_root / "posterior_predictive" / "truth" / "default"
        experiment_row = {
            "experiment_name": "exp_a",
            "experiment_slug": "exp-a",
            "descriptor": "demo",
            "experiment_path": str(experiment_root),
            "intervention_source": "generated",
            "graph_source": "erdos_renyi",
        }
        panel_context = {"N": 6, "T": 4, "s": 1}
        target = {
            "source_type": "truth",
            "source_name": "truth",
            "source_slug": "truth",
            "intervention_source": "observed_experiment",
            "intervention_name": "observed_experiment",
            "intervention_slug": "observed_experiment",
        }
        run_spec = {"name": "default", "slug": "default"}
        summary = {
            "s": 1,
            "mean_abs_zscore": 0.25,
            "max_abs_zscore": 0.5,
            "coverage_rate": 0.8,
            "num_statistics": 10,
        }

        expected_row = build_manifest_row(
            experiment_row=experiment_row,
            panel_context=panel_context,
            target=target,
            run_spec=run_spec,
            latent_rank=2,
            num_samples=8,
            gibbs_sweeps=100,
            seed=17,
            output_root=output_root,
            summary=summary,
        )
        metadata = {
            "run_name": "default",
            "run_slug": "default",
            "source_type": "truth",
            "source_name": "truth",
            "source_slug": "truth",
            "intervention_source": "observed_experiment",
            "intervention_name": "observed_experiment",
            "intervention_slug": "observed_experiment",
            "latent_rank": 2,
            "num_samples": 8,
            "gibbs_sweeps": 100,
            "seed": 17,
            "s": 1,
            "num_units": 6,
            "num_time_steps": 4,
            "summary": summary,
        }

        reconstructed_row = manifest_row_from_metadata(
            experiment_row,
            metadata,
            output_root,
        )

        self.assertEqual(expected_row, reconstructed_row)

    def test_build_manifest_row_matches_metadata_reconstruction_for_counterfactual(
        self,
    ) -> None:
        experiment_root = self.root / "manifest_counterfactual"
        output_root = (
            experiment_root
            / "counterfactual"
            / "fit_rank_0"
            / "full_on_from_s"
            / "default"
        )
        experiment_row = {
            "experiment_name": "exp_a",
            "experiment_slug": "exp-a",
            "descriptor": "demo",
            "experiment_path": str(experiment_root),
            "intervention_source": "generated",
            "graph_source": "erdos_renyi",
        }
        panel_context = {"N": 6, "T": 4, "s": 2}
        target = {
            "source_type": "fit",
            "source_name": "rank_0",
            "source_slug": "fit_rank_0",
            "intervention_source": "saved_intervention",
            "intervention_name": "full_on_from_s",
            "intervention_slug": "full_on_from_s",
        }
        run_spec = {"name": "default", "slug": "default"}
        summary = {"s": 2, "num_samples": 8, "num_units": 6}

        expected_row = build_manifest_row(
            experiment_row=experiment_row,
            panel_context=panel_context,
            target=target,
            run_spec=run_spec,
            latent_rank=3,
            num_samples=8,
            gibbs_sweeps=100,
            seed=21,
            output_root=output_root,
            summary=summary,
        )
        metadata = {
            "run_name": "default",
            "run_slug": "default",
            "source_type": "fit",
            "source_name": "rank_0",
            "source_slug": "fit_rank_0",
            "intervention_source": "saved_intervention",
            "intervention_name": "full_on_from_s",
            "intervention_slug": "full_on_from_s",
            "latent_rank": 3,
            "num_samples": 8,
            "gibbs_sweeps": 100,
            "seed": 21,
            "s": 2,
            "num_units": 6,
            "num_time_steps": 4,
            "summary": summary,
        }

        reconstructed_row = manifest_row_from_metadata(
            experiment_row,
            metadata,
            output_root,
        )

        self.assertEqual(expected_row, reconstructed_row)
        self.assertEqual(reconstructed_row["mean_abs_zscore"], "")

    def test_write_observed_predictive_summary_tables_preserves_contract(self) -> None:
        output_root = self.root / "observed_writer"
        sample_summaries = {
            "overall_mean_magnetization": np.asarray([0.0, 0.25], dtype=float),
            "post_intervention_mean_magnetization": np.asarray([0.5, 0.75], dtype=float),
            "unit_mean_magnetization": np.asarray(
                [[0.0, 0.5], [0.25, 0.75]],
                dtype=float,
            ),
            "time_mean_magnetization": np.asarray(
                [[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]],
                dtype=float,
            ),
        }
        mean_rows = [
            {
                "statistic": "overall_mean_magnetization",
                "observed_value": 0.1,
                "sample_mean": 0.125,
                "sample_std": 0.125,
                "abs_error": 0.025,
                "q025": 0.0,
                "q500": 0.125,
                "q975": 0.25,
                "in_95_interval": True,
                "num_finite_samples": 2,
            }
        ]
        unit_rows = [
            {
                "unit_index": 0,
                "observed_value": 0.2,
                "sample_mean": 0.125,
                "sample_std": 0.125,
                "abs_error": 0.075,
                "squared_error": 0.005625,
                "q025": 0.0,
                "q500": 0.125,
                "q975": 0.25,
                "in_95_interval": True,
                "num_finite_samples": 2,
            }
        ]
        time_rows = [
            {
                "time_index": 0,
                "observed_value": 0.1,
                "sample_mean": 0.125,
                "sample_std": 0.125,
                "abs_error": 0.025,
                "squared_error": 0.000625,
                "q025": 0.0,
                "q500": 0.125,
                "q975": 0.25,
                "in_95_interval": True,
                "num_finite_samples": 2,
            }
        ]

        sample_npz_path, mean_csv_path, unit_csv_path, time_csv_path = (
            write_observed_predictive_summary_tables(
                output_root,
                sample_summaries=sample_summaries,
                mean_rows=mean_rows,
                unit_rows=unit_rows,
                time_rows=time_rows,
            )
        )

        self.assertTrue(sample_npz_path.exists())
        self.assertTrue(mean_csv_path.exists())
        self.assertTrue(unit_csv_path.exists())
        self.assertTrue(time_csv_path.exists())
        with np.load(io_path(sample_npz_path), allow_pickle=False) as data:
            self.assertEqual(
                np.asarray(data["unit_mean_magnetization"], dtype=float).shape,
                (2, 2),
            )
        with mean_csv_path.open("r", encoding="utf-8", newline="") as handle:
            mean_reader = csv.DictReader(handle)
            self.assertEqual(
                mean_reader.fieldnames,
                [
                    "statistic",
                    "observed_value",
                    "sample_mean",
                    "sample_std",
                    "abs_error",
                    "q025",
                    "q500",
                    "q975",
                    "in_95_interval",
                    "num_finite_samples",
                ],
            )

    def test_write_counterfactual_summary_tables_preserves_contract(self) -> None:
        output_root = self.root / "counterfactual_writer"
        sample_summaries = {
            "overall_mean_magnetization": np.asarray([0.0, 0.25], dtype=float),
            "post_intervention_mean_magnetization": np.asarray([0.5, 0.75], dtype=float),
            "unit_mean_magnetization": np.asarray(
                [[0.0, 0.5], [0.25, 0.75]],
                dtype=float,
            ),
            "time_mean_magnetization": np.asarray(
                [[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]],
                dtype=float,
            ),
        }

        sample_npz_path, summary_csv_path, unit_csv_path, time_csv_path = (
            write_counterfactual_summary_tables(
                output_root,
                sample_summaries=sample_summaries,
            )
        )

        self.assertTrue(sample_npz_path.exists())
        self.assertTrue(summary_csv_path.exists())
        self.assertTrue(unit_csv_path.exists())
        self.assertTrue(time_csv_path.exists())
        with np.load(io_path(sample_npz_path), allow_pickle=False) as data:
            self.assertEqual(
                np.asarray(data["time_mean_magnetization"], dtype=float).shape,
                (2, 3),
            )
        with summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
            summary_reader = csv.DictReader(handle)
            self.assertEqual(
                summary_reader.fieldnames,
                [
                    "statistic",
                    "sample_mean",
                    "sample_std",
                    "q025",
                    "q500",
                    "q975",
                    "num_finite_samples",
                ],
            )

    def test_simulate_outcomes_for_bundle_ignores_beta_mask_flags(self) -> None:
        """Beta masking is only for MPLE fitting loss, not for sampling."""
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
            beta_mask_post_e=False,
            latent_rank=0,
            t_steps=2,
            field_matrix=field_matrix,
            gamma_matrix=gamma_matrix,
        )
        unmasked_bundle = OutcomeParameterBundle(
            source_type="fit",
            source_name="rank_0",
            beta=12.0,
            xi=0.0,
            eta=0.0,
            beta_mask_pre_s=False,
            beta_mask_post_e=False,
            latent_rank=0,
            t_steps=2,
            field_matrix=field_matrix,
            gamma_matrix=gamma_matrix,
        )

        masked_result = simulate_outcomes_for_bundle(
            masked_bundle,
            x_0=x_0,
            z=z,
            gibbs_sweeps=1,
            seed=0,
        )
        unmasked_result = simulate_outcomes_for_bundle(
            unmasked_bundle,
            x_0=x_0,
            z=z,
            gibbs_sweeps=1,
            seed=0,
        )

        self.assertTrue(np.array_equal(masked_result, unmasked_result))

    def test_validation_metric_utils_sampler_clamps_non_validation_and_is_reproducible(
        self,
    ) -> None:
        panel_context = {
            "x": np.asarray([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], dtype=float),
            "z": np.asarray([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=float),
            "x_0": np.asarray([1.0, -1.0, 1.0], dtype=float),
            "s": 1,
            "e": 2,
        }
        bundle = OutcomeParameterBundle(
            source_type="fit",
            source_name="rank_0",
            beta=1.2,
            xi=0.0,
            eta=0.0,
            beta_mask_pre_s=False,
            beta_mask_post_e=False,
            latent_rank=0,
            t_steps=2,
            field_matrix=np.zeros((2, 3), dtype=float),
            gamma_matrix=np.zeros((3, 3), dtype=float),
        )
        validation_mask = np.asarray(
            [[False, True, False], [True, False, False]],
            dtype=bool,
        )

        sampled_one = validation_metrics.sample_validation_panel_conditional(
            panel_context=panel_context,
            bundle=bundle,
            validation_loss_mask=validation_mask,
            gibbs_sweeps=2,
            seed=17,
        )
        sampled_two = validation_metrics.sample_validation_panel_conditional(
            panel_context=panel_context,
            bundle=bundle,
            validation_loss_mask=validation_mask,
            gibbs_sweeps=2,
            seed=17,
        )

        self.assertTrue(np.array_equal(sampled_one, sampled_two))
        self.assertTrue(
            np.array_equal(
                sampled_one[~validation_mask],
                np.asarray(panel_context["x"], dtype=float)[~validation_mask],
            )
        )

    def test_validation_metric_utils_sampler_does_not_warm_start_validation_nodes(
        self,
    ) -> None:
        panel_context = {
            "x": np.ones((2, 8), dtype=float),
            "z": np.zeros((2, 8), dtype=float),
            "x_0": -np.ones(8, dtype=float),
            "s": 1,
            "e": 2,
        }
        bundle = OutcomeParameterBundle(
            source_type="fit",
            source_name="rank_0",
            beta=0.0,
            xi=0.0,
            eta=0.0,
            beta_mask_pre_s=False,
            beta_mask_post_e=False,
            latent_rank=0,
            t_steps=2,
            field_matrix=np.zeros((2, 8), dtype=float),
            gamma_matrix=np.zeros((8, 8), dtype=float),
        )
        validation_mask = np.ones((2, 8), dtype=bool)

        sampled_x = validation_metrics.sample_validation_panel_conditional(
            panel_context=panel_context,
            bundle=bundle,
            validation_loss_mask=validation_mask,
            gibbs_sweeps=0,
            seed=0,
        )

        self.assertFalse(np.array_equal(sampled_x, np.asarray(panel_context["x"], dtype=float)))

    def test_validation_metric_utils_full_panel_sampler_can_change_non_test_entries(
        self,
    ) -> None:
        panel_context = {
            "x": np.ones((2, 8), dtype=float),
            "z": np.zeros((2, 8), dtype=float),
            "x_0": -np.ones(8, dtype=float),
            "s": 1,
            "e": 2,
        }
        bundle = OutcomeParameterBundle(
            source_type="fit",
            source_name="rank_0",
            beta=0.0,
            xi=0.0,
            eta=0.0,
            beta_mask_pre_s=False,
            beta_mask_post_e=False,
            latent_rank=0,
            t_steps=2,
            field_matrix=np.zeros((2, 8), dtype=float),
            gamma_matrix=np.zeros((8, 8), dtype=float),
        )
        test_mask = np.asarray(
            [[True, False, False, False, False, False, False, False], [False] * 8],
            dtype=bool,
        )

        sampled_x = validation_metrics.sample_full_panel_regeneration(
            panel_context=panel_context,
            bundle=bundle,
            gibbs_sweeps=0,
            seed=0,
        )

        observed_x = np.asarray(panel_context["x"], dtype=float)
        self.assertFalse(np.array_equal(sampled_x, observed_x))
        self.assertTrue(np.any(sampled_x[~test_mask] != observed_x[~test_mask]))

    def test_validation_metric_utils_reports_full_panel_regeneration_buckets(
        self,
    ) -> None:
        panel_context = {
            "x": np.asarray([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]], dtype=float),
            "z": np.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=float),
            "x_0": np.asarray([1.0, -1.0], dtype=float),
            "s": 1,
            "e": 3,
        }
        bundle = OutcomeParameterBundle(
            source_type="fit",
            source_name="rank_0",
            beta=0.0,
            xi=0.0,
            eta=0.0,
            beta_mask_pre_s=False,
            beta_mask_post_e=False,
            latent_rank=0,
            t_steps=3,
            field_matrix=np.zeros((3, 2), dtype=float),
            gamma_matrix=np.zeros((2, 2), dtype=float),
        )
        training_mask = np.asarray(
            [[True, False], [False, False], [False, False]],
            dtype=bool,
        )
        test_mask = np.asarray(
            [[False, True], [True, False], [False, False]],
            dtype=bool,
        )

        with mock.patch.object(
            validation_metrics,
            "sample_full_panel_regeneration",
            wraps=validation_metrics.sample_full_panel_regeneration,
        ) as sampler:
            metrics = validation_metrics.evaluate_test_metrics(
                panel_context=panel_context,
                bundle=bundle,
                training_loss_mask=training_mask,
                test_loss_mask=test_mask,
                sampling={"num_samples": 3, "gibbs_sweeps": 0, "seed": 5},
            )

        self.assertEqual(sampler.call_count, 3)
        self.assertIn("full_panel_all_mean_magnetization_abs_diff", metrics)
        self.assertIn("full_panel_test_mean_magnetization_abs_diff", metrics)
        self.assertIn("full_panel_separator_mean_magnetization_abs_diff", metrics)
        self.assertIn("full_panel_treated_test_mean_magnetization_abs_diff", metrics)
        self.assertIn("full_panel_untreated_test_mean_magnetization_abs_diff", metrics)
        self.assertEqual(metrics["full_panel_num_all_slots"], 6)
        self.assertEqual(metrics["full_panel_num_all_post_s_slots"], 4)
        self.assertEqual(metrics["full_panel_num_training_slots"], 1)
        self.assertEqual(metrics["full_panel_num_training_post_s_slots"], 0)
        self.assertEqual(metrics["full_panel_num_separator_slots"], 3)
        self.assertEqual(metrics["full_panel_num_separator_post_s_slots"], 3)
        self.assertEqual(metrics["full_panel_num_test_slots"], 2)
        self.assertEqual(metrics["full_panel_num_test_post_s_slots"], 1)
        self.assertEqual(metrics["full_panel_num_treated_test_slots"], 2)
        self.assertEqual(metrics["full_panel_num_treated_test_post_s_slots"], 1)
        self.assertEqual(metrics["full_panel_num_untreated_test_slots"], 0)
        self.assertEqual(metrics["full_panel_num_untreated_test_post_s_slots"], 0)
        self.assertIsNone(metrics["full_panel_training_post_s_mean_magnetization_abs_diff"])
        self.assertIsNone(metrics["full_panel_untreated_test_mean_magnetization_abs_diff"])
        self.assertIsNone(
            metrics["full_panel_untreated_test_post_s_mean_magnetization_abs_diff"]
        )
        self.assertAlmostEqual(
            float(metrics["full_panel_all_observed_mean_magnetization"]),
            float(np.mean(np.asarray(panel_context["x"], dtype=float))),
            places=12,
        )

    def test_validation_metric_utils_reports_magnetization_metrics(self) -> None:
        panel_context = {
            "x": np.asarray([[1.0, -1.0], [1.0, 1.0]], dtype=float),
            "z": np.zeros((2, 2), dtype=float),
            "x_0": np.asarray([1.0, -1.0], dtype=float),
            "s": 1,
            "e": 2,
        }
        bundle = OutcomeParameterBundle(
            source_type="fit",
            source_name="rank_0",
            beta=0.0,
            xi=0.0,
            eta=0.0,
            beta_mask_pre_s=False,
            beta_mask_post_e=False,
            latent_rank=0,
            t_steps=2,
            field_matrix=np.zeros((2, 2), dtype=float),
            gamma_matrix=np.zeros((2, 2), dtype=float),
        )
        training_mask = np.asarray([[True, False], [False, False]], dtype=bool)
        validation_mask = np.asarray([[False, True], [True, False]], dtype=bool)

        metrics = validation_metrics.evaluate_fold_metrics(
            panel_context=panel_context,
            bundle=bundle,
            training_loss_mask=training_mask,
            validation_loss_mask=validation_mask,
            validation_sampling={"num_samples": 3, "gibbs_sweeps": 1, "seed": 5},
        )

        self.assertIn("validation_mean_magnetization_abs_diff", metrics)
        self.assertIn("post_s_validation_mean_magnetization_abs_diff", metrics)
        self.assertGreaterEqual(float(metrics["validation_mean_magnetization_abs_diff"]), 0.0)
        self.assertAlmostEqual(
            float(metrics["validation_observed_mean_magnetization"]),
            float(np.mean(np.asarray(panel_context["x"], dtype=float)[validation_mask])),
            places=12,
        )

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
            time_means=[0.05, 0.30, 0.55, -0.10],
            time_q025=[-0.05, 0.20, 0.45, -0.20],
            time_q975=[0.15, 0.40, 0.65, 0.00],
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
            time_means=[0.08, 0.28, 0.50, -0.06],
            time_q025=[-0.02, 0.18, 0.40, -0.16],
            time_q975=[0.18, 0.38, 0.60, 0.04],
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
            time_means=[0.45, -0.10, 0.80, 0.25],
            time_q025=[0.35, -0.20, 0.70, 0.15],
            time_q975=[0.55, 0.00, 0.90, 0.35],
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
        self.assertIn("truth_time_mean_squared_error_mean", rows[0])
        self.assertEqual(float(by_name["truth"]["truth_unit_mean_squared_error_mean"]), 0.0)
        self.assertEqual(float(by_name["truth"]["truth_time_mean_squared_error_mean"]), 0.0)
        self.assertEqual(float(by_name["truth"]["truth_overall_mean_magnetization_abs_error"]), 0.0)
        self.assertEqual(by_name["truth"]["truth_rank_in_run"], "")
        self.assertEqual(by_name["truth"]["truth_is_best"], "")

        better_mse = float(by_name["better_fit"]["truth_unit_mean_squared_error_mean"])
        worse_mse = float(by_name["worse_fit"]["truth_unit_mean_squared_error_mean"])
        better_time_mse = float(by_name["better_fit"]["truth_time_mean_squared_error_mean"])
        worse_time_mse = float(by_name["worse_fit"]["truth_time_mean_squared_error_mean"])
        self.assertLess(better_mse, worse_mse)
        self.assertLess(better_time_mse, worse_time_mse)
        self.assertEqual(by_name["better_fit"]["truth_rank_in_run"], "1")
        self.assertEqual(by_name["better_fit"]["truth_is_best"], "True")
        self.assertEqual(by_name["worse_fit"]["truth_rank_in_run"], "2")
        self.assertEqual(by_name["worse_fit"]["truth_is_best"], "False")
        self.assertGreater(
            float(by_name["better_fit"]["truth_unit_mean_95_interval_coverage_rate"]),
            float(by_name["worse_fit"]["truth_unit_mean_95_interval_coverage_rate"]),
        )
        self.assertGreater(
            float(by_name["better_fit"]["truth_time_mean_95_interval_coverage_rate"]),
            float(by_name["worse_fit"]["truth_time_mean_95_interval_coverage_rate"]),
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
            time_means=[0.10, 0.25, 0.05],
            time_q025=[0.0, 0.15, -0.05],
            time_q975=[0.20, 0.35, 0.15],
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
        self.assertEqual(row["truth_time_mean_squared_error_mean"], "")
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
        self.assertEqual(row["truth_time_mean_squared_error_mean"], "")
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
                    "    scalars:",
                    "      beta: 0.2",
                    "      xi: 0.1",
                    "      eta: 0.05",
                    "experiments:",
                    "  - name: smoke_rank_0",
                    "  - name: smoke_rank_1",
                ]
            ),
            encoding="utf-8",
        )
        intervention_spec_path.write_text(
            "\n".join(
                [
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

        self.assertEqual(
            Path(library_manifest),
            self.root / "generated" / "intervention_library_manifest.csv",
        )
        manifest_rows = read_csv_manifest(library_manifest)
        self.assertEqual(len(manifest_rows), 6)
        self.assertEqual(
            {row["experiment_name"] for row in manifest_rows},
            {"smoke_rank_0", "smoke_rank_1"},
        )
        for experiment_name in ("smoke_rank_0", "smoke_rank_1"):
            experiment_root = self.root / "generated" / experiment_name
            observed_copy = load_saved_intervention_context(
                experiment_root, "observed_copy"
            )
            full_on = load_saved_intervention_context(
                experiment_root, "full_on_from_s"
            )
            single_unit = load_saved_intervention_context(
                experiment_root, "single_unit_2_from_step_2"
            )

            self.assertTrue(np.array_equal(observed_copy.z_0, np.zeros(6, dtype=float)))
            self.assertTrue(np.array_equal(full_on.z, np.ones((4, 6), dtype=float)))
            self.assertEqual(full_on.s, 0)
            self.assertTrue(
                np.array_equal(single_unit.z[:2, 2], -np.ones(2, dtype=float))
            )
            self.assertTrue(
                np.array_equal(single_unit.z[2:, 2], np.ones(2, dtype=float))
            )
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
                    "    scalars:",
                    "      beta: 0.2",
                    "      xi: 0.1",
                    "      eta: 0.05",
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
        observed_root = experiment_root / "posterior_predictive" / "truth" / "default"
        observed_output = observed_root / "posterior_predictive_stats.csv"
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
        self.assertTrue((observed_root / "posterior_predictive_metadata.yaml").exists())
        self.assertTrue(
            (observed_root / "posterior_predictive_sample_summaries.npz").exists()
        )
        self.assertTrue(
            (observed_root / "posterior_predictive_mean_summary.csv").exists()
        )
        self.assertTrue(
            (observed_root / "posterior_predictive_unit_summary.csv").exists()
        )
        self.assertTrue(
            (observed_root / "posterior_predictive_time_summary.csv").exists()
        )
        self.assertTrue(predictive_manifest.exists())
        self.assertTrue((counterfactual_root / "counterfactual_metadata.yaml").exists())
        self.assertTrue(
            (counterfactual_root / "counterfactual_sample_summaries.npz").exists()
        )
        self.assertTrue((counterfactual_root / "counterfactual_summary.csv").exists())
        self.assertTrue(
            (counterfactual_root / "counterfactual_unit_summary.csv").exists()
        )
        self.assertTrue(
            (counterfactual_root / "counterfactual_time_summary.csv").exists()
        )
        self.assertFalse(
            (counterfactual_root / "posterior_predictive_stats.csv").exists()
        )

        observed_panel = load_experiment_panel_context(experiment_root)
        with np.load(
            io_path(observed_root / "posterior_predictive_sample_summaries.npz"),
            allow_pickle=False,
        ) as data:
            self.assertEqual(
                np.asarray(data["overall_mean_magnetization"], dtype=float).shape,
                (4,),
            )
            self.assertEqual(
                np.asarray(data["post_intervention_mean_magnetization"], dtype=float).shape,
                (4,),
            )
            self.assertEqual(
                np.asarray(data["unit_mean_magnetization"], dtype=float).shape,
                (4, 6),
            )
            self.assertEqual(
                np.asarray(data["time_mean_magnetization"], dtype=float).shape,
                (4, 4),
            )

        with (observed_root / "posterior_predictive_mean_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            mean_rows = {row["statistic"]: row for row in csv.DictReader(handle)}
        self.assertAlmostEqual(
            float(mean_rows["overall_mean_magnetization"]["observed_value"]),
            float(np.mean(np.asarray(observed_panel["x"], dtype=float))),
        )
        self.assertAlmostEqual(
            float(mean_rows["post_intervention_mean_magnetization"]["observed_value"]),
            float(
                np.mean(
                    np.asarray(observed_panel["x"], dtype=float)[
                        int(observed_panel["s"]) :, :
                    ]
                )
            ),
        )

        with (observed_root / "posterior_predictive_unit_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            unit_rows = list(csv.DictReader(handle))
        with (observed_root / "posterior_predictive_time_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            time_rows = list(csv.DictReader(handle))
        self.assertEqual(len(unit_rows), int(observed_panel["N"]))
        self.assertEqual(len(time_rows), int(observed_panel["T"]))
        self.assertAlmostEqual(
            float(unit_rows[0]["observed_value"]),
            float(np.mean(np.asarray(observed_panel["x"], dtype=float), axis=0)[0]),
        )
        self.assertAlmostEqual(
            float(time_rows[0]["observed_value"]),
            float(np.mean(np.asarray(observed_panel["x"], dtype=float), axis=1)[0]),
        )

        observed_metadata = OmegaConf.to_container(
            OmegaConf.load(io_path(observed_root / "posterior_predictive_metadata.yaml")),
            resolve=True,
        )
        self.assertIsInstance(observed_metadata, dict)
        observed_summary = dict(observed_metadata["summary"])
        self.assertAlmostEqual(
            float(observed_summary["overall_mean_abs_error"]),
            float(mean_rows["overall_mean_magnetization"]["abs_error"]),
        )
        self.assertAlmostEqual(
            float(observed_summary["post_intervention_mean_abs_error"]),
            float(mean_rows["post_intervention_mean_magnetization"]["abs_error"]),
        )
        self.assertAlmostEqual(
            float(observed_summary["unit_mean_abs_error_mean"]),
            float(np.mean([float(row["abs_error"]) for row in unit_rows])),
        )
        self.assertAlmostEqual(
            float(observed_summary["unit_mean_rmse"]),
            float(
                np.sqrt(
                    np.mean([float(row["squared_error"]) for row in unit_rows])
                )
            ),
        )
        self.assertAlmostEqual(
            float(observed_summary["time_mean_abs_error_mean"]),
            float(np.mean([float(row["abs_error"]) for row in time_rows])),
        )
        self.assertAlmostEqual(
            float(observed_summary["time_mean_rmse"]),
            float(
                np.sqrt(
                    np.mean([float(row["squared_error"]) for row in time_rows])
                )
            ),
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
                    "    scalars:",
                    "      beta: 0.2",
                    "      xi: 0.1",
                    "      eta: 0.05",
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


class GraphPartitioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "experiments" / f".tmp_graph_partition_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_experiment_root(
        self,
        gamma_matrix: np.ndarray,
        *,
        use_sparse_artifact: bool = False,
        include_node_index: bool = False,
        t_steps: int = 10,
        include_time_index: bool = False,
    ) -> Path:
        experiment_root = self.root / f"experiment_{uuid.uuid4().hex[:8]}"
        experiment_root.mkdir(parents=True, exist_ok=True)
        if use_sparse_artifact:
            sparse.save_npz(
                experiment_root / "gamma_matrix_sparse.npz",
                sparse.csr_matrix(np.asarray(gamma_matrix, dtype=float)),
            )
        else:
            np.save(experiment_root / "gamma_matrix.npy", np.asarray(gamma_matrix, dtype=float))
        n_nodes = int(gamma_matrix.shape[0])
        np.savez(
            experiment_root / "panel_data.npz",
            x=np.zeros((int(t_steps), n_nodes), dtype=float),
            z=np.zeros((int(t_steps), n_nodes), dtype=float),
        )
        np.save(experiment_root / "x_0.npy", np.zeros(n_nodes, dtype=float))
        np.save(experiment_root / "z_0.npy", np.zeros(n_nodes, dtype=float))
        if include_node_index:
            with (experiment_root / "node_index.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["fips", "county_name"])
                writer.writeheader()
                for index in range(int(gamma_matrix.shape[0])):
                    writer.writerow(
                        {
                            "fips": f"{index:05d}",
                            "county_name": f"county_{index}",
                        }
                    )
        if include_time_index:
            with (experiment_root / "time_index.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["WeekStartDate", "WeekEndDate", "iso_week"],
                )
                writer.writeheader()
                for time_index in range(int(t_steps)):
                    writer.writerow(
                        {
                            "WeekStartDate": f"2021-01-{time_index + 1:02d}",
                            "WeekEndDate": f"2021-01-{time_index + 2:02d}",
                            "iso_week": str(time_index + 1),
                        }
                    )
        return experiment_root

    def _recompute_partition_metrics(
        self,
        gamma_matrix: np.ndarray,
        fold_ids: list[int],
    ) -> tuple[int, int, list[list[int]]]:
        adjacency, _ = cv_folds._support_adjacency_from_gamma(gamma_matrix)
        membership = np.asarray(fold_ids, dtype=int)
        metrics = cv_folds._compute_partition_metrics(
            adjacency,
            membership,
            num_folds=max(fold_ids) + 1,
        )
        separator_sets = [
            sorted(int(value) for value in values)
            for values in metrics["separator_sets"]
        ]
        return (
            int(metrics["cut_edge_count"]),
            int(metrics["separator_union_vertex_count"]),
            separator_sets,
        )

    def _write_generation_manifest(self, experiment_roots: list[Path]) -> Path:
        manifest_path = self.root / f"generation_manifest_{uuid.uuid4().hex[:8]}.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["experiment_name", "experiment_path"],
            )
            writer.writeheader()
            for index, experiment_root in enumerate(experiment_roots, start=1):
                writer.writerow(
                    {
                        "experiment_name": f"exp_{index}",
                        "experiment_path": str(experiment_root.resolve()),
                    }
                )
        return manifest_path

    def test_build_cv_folds_writes_dense_outputs_and_metrics(self) -> None:
        gamma_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        experiment_root = self._write_experiment_root(
            gamma_matrix,
            include_node_index=True,
            include_time_index=True,
            t_steps=10,
        )

        class FakePyMetis:
            @staticmethod
            def part_graph(nparts, adjacency=None, recursive=None, contiguous=None):
                self.assertEqual(nparts, 5)
                self.assertFalse(bool(recursive))
                self.assertTrue(bool(contiguous))
                self.assertEqual(len(adjacency), 5)
                return 99, [0, 1, 2, 3, 4]

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=FakePyMetis(),
        ):
            output_root = cv_folds._run_build_cv_folds_for_experiment(
                experiment_root,
                num_folds=5,
                seed=11,
                contiguous=True,
            )

        bundle_metadata = OmegaConf.to_container(
            OmegaConf.load(output_root / "bundle_metadata.yaml"),
            resolve=True,
        )

        self.assertTrue(os.path.exists(io_path(output_root / "bundle_metadata.yaml")))
        self.assertTrue(os.path.exists(io_path(output_root / "model_selection_folds.npz")))
        self.assertTrue(os.path.exists(io_path(output_root / "fold_summary.csv")))

    def test_build_cv_folds_writes_sparse_artifact_metadata(self) -> None:
        gamma_matrix = np.array(
            [
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        experiment_root = self._write_experiment_root(
            gamma_matrix,
            use_sparse_artifact=True,
        )

        class FakePyMetis:
            @staticmethod
            def part_graph(nparts, adjacency=None, recursive=None, contiguous=None):
                self.assertEqual(nparts, 5)
                self.assertTrue(bool(recursive))
                return 7, [0, 0, 1, 3, 4]

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=FakePyMetis(),
        ):
            output_root = cv_folds._run_build_cv_folds_for_experiment(
                experiment_root,
                num_folds=5,
                recursive=True,
            )

        bundle_metadata = OmegaConf.to_container(
            OmegaConf.load(output_root / "bundle_metadata.yaml"),
            resolve=True,
        )
        self.assertTrue(os.path.exists(io_path(output_root / "bundle_metadata.yaml")))
        self.assertTrue(os.path.exists(io_path(output_root / "model_selection_folds.npz")))
        self.assertTrue(os.path.exists(io_path(output_root / "fold_summary.csv")))

    def test_build_cv_folds_rejects_non_square_gamma(self) -> None:
        experiment_root = self._write_experiment_root(np.zeros((2, 3), dtype=float))
        with self.assertRaisesRegex(ValueError, "must be square"):
            cv_folds._run_build_cv_folds_for_experiment(
                experiment_root,
                num_folds=5,
            )

    def test_build_cv_folds_rejects_asymmetric_gamma(self) -> None:
        experiment_root = self._write_experiment_root(
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=float,
            )
        )
        with self.assertRaisesRegex(ValueError, "must be symmetric"):
            cv_folds._run_build_cv_folds_for_experiment(
                experiment_root,
                num_folds=5,
            )

    def test_build_cv_folds_rejects_nonzero_diagonal(self) -> None:
        experiment_root = self._write_experiment_root(
            np.eye(5, dtype=float)
        )
        with self.assertRaisesRegex(ValueError, "zero diagonal"):
            cv_folds._run_build_cv_folds_for_experiment(
                experiment_root,
                num_folds=5,
            )

    def test_build_time_block_plan_handles_9_10_and_uneven_horizons(self) -> None:
        plan_9 = cv_folds._build_time_block_plan(9, num_folds=5)
        plan_10 = cv_folds._build_time_block_plan(10, num_folds=5)
        plan_13 = cv_folds._build_time_block_plan(13, num_folds=5)

        self.assertEqual(plan_9["block_sizes"], [1, 2, 2, 2, 2])
        self.assertEqual(plan_10["block_sizes"], [2, 2, 2, 2, 2])
        self.assertEqual(plan_13["block_sizes"], [3, 3, 3, 2, 2])
        self.assertEqual(plan_9["transition_time_indices"], [1, 3, 5, 7])
        self.assertEqual(plan_10["transition_time_indices"], [2, 4, 6, 8])
        self.assertEqual(plan_13["transition_time_indices"], [3, 6, 9, 11])

    def test_build_cv_folds_rejects_too_short_time_horizon(self) -> None:
        experiment_root = self._write_experiment_root(
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            ),
            t_steps=8,
        )
        with self.assertRaisesRegex(ValueError, "at least 9 time steps"):
            cv_folds._run_build_cv_folds_for_experiment(
                experiment_root,
                num_folds=5,
            )

    def test_build_cv_folds_fails_cleanly_without_pymetis(self) -> None:
        experiment_root = self._write_experiment_root(
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            )
        )
        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            side_effect=RuntimeError("pymetis is required for CV graph partitioning"),
        ):
            with self.assertRaisesRegex(RuntimeError, "pymetis is required"):
                cv_folds._run_build_cv_folds_for_experiment(
                    experiment_root,
                    num_folds=5,
                )

    def test_build_cv_folds_falls_back_to_inferred_integer_time_index(self) -> None:
        gamma_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        experiment_root = self._write_experiment_root(gamma_matrix, include_time_index=False)

        class FakePyMetis:
            @staticmethod
            def part_graph(nparts, adjacency=None, recursive=None, contiguous=None):
                return 4, [0, 1, 2, 3, 4]

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=FakePyMetis(),
        ):
            output_root = cv_folds._run_build_cv_folds_for_experiment(
                experiment_root,
                num_folds=5,
            )

        bundle_metadata = OmegaConf.to_container(
            OmegaConf.load(output_root / "bundle_metadata.yaml"),
            resolve=True,
        )
        self.assertTrue(os.path.exists(io_path(output_root / "bundle_metadata.yaml")))
        self.assertTrue(os.path.exists(io_path(output_root / "model_selection_folds.npz")))
        self.assertTrue(os.path.exists(io_path(output_root / "fold_summary.csv")))

    def test_markov_blanket_summary_detects_spatial_violation(self) -> None:
        adjacency = [
            [1],
            [0],
        ]
        role_codes = np.array(
            [
                [
                    [cv_folds.ROLE_CODE_VALIDATION, cv_folds.ROLE_CODE_TRAINING],
                    [cv_folds.ROLE_CODE_SEPARATOR, cv_folds.ROLE_CODE_SEPARATOR],
                ]
            ],
            dtype=int,
        )
        summary = cv_folds._summarize_markov_blanket_validation(adjacency, role_codes)

        self.assertFalse(summary["blanket_validation_passed"])
        self.assertEqual(summary["spatial_violation_edge_count"], 1)
        self.assertEqual(summary["temporal_violation_edge_count"], 0)
        self.assertEqual(summary["violations_by_fold"], [1])

    def test_markov_blanket_summary_detects_temporal_violation(self) -> None:
        adjacency = [
            [],
        ]
        role_codes = np.array(
            [
                [
                    [cv_folds.ROLE_CODE_VALIDATION],
                    [cv_folds.ROLE_CODE_TRAINING],
                ]
            ],
            dtype=int,
        )
        summary = cv_folds._summarize_markov_blanket_validation(adjacency, role_codes)

        self.assertFalse(summary["blanket_validation_passed"])
        self.assertEqual(summary["spatial_violation_edge_count"], 0)
        self.assertEqual(summary["temporal_violation_edge_count"], 1)
        self.assertEqual(summary["violations_by_fold"], [1])

    def test_build_cv_folds_is_stable_for_repeated_runs(self) -> None:
        np.random.seed(42)
        n_nodes = 1000
        n_folds = 5
        nodes_per_fold = n_nodes // n_folds

        gamma_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
        for fold_i in range(n_folds):
            for fold_j in range(max(0, fold_i - 1), min(n_folds, fold_i + 2)):
                start_i = fold_i * nodes_per_fold
                end_i = start_i + nodes_per_fold
                start_j = fold_j * nodes_per_fold
                end_j = start_j + nodes_per_fold
                p_edge = 0.01
                block = (np.random.random((nodes_per_fold, nodes_per_fold)) < p_edge).astype(float)
                gamma_matrix[start_i:end_i, start_j:end_j] = block

        gamma_matrix = np.triu(gamma_matrix) + np.triu(gamma_matrix, k=1).T
        np.fill_diagonal(gamma_matrix, 0.0)
        experiment_root = self._write_experiment_root(gamma_matrix)

        class FakePyMetis:
            @staticmethod
            def part_graph(nparts, adjacency=None, recursive=None, contiguous=None):
                n = len(adjacency) if adjacency is not None else 1000
                nodes_per_part = n // nparts
                return 4, [min(i // nodes_per_part, nparts - 1) for i in range(n)]

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=FakePyMetis(),
        ):
            first_output = cv_folds._run_build_cv_folds_for_experiment(
                experiment_root,
                num_folds=5,
                seed=3,
            )
            second_output = cv_folds._run_build_cv_folds_for_experiment(
                experiment_root,
                num_folds=5,
                seed=3,
            )

        first_metadata = (first_output / "bundle_metadata.yaml").read_text(encoding="utf-8")
        second_metadata = (second_output / "bundle_metadata.yaml").read_text(encoding="utf-8")
        with np.load(io_path(first_output / "model_selection_folds.npz"), allow_pickle=False) as data:
            first_training = np.asarray(data["training_masks"], dtype=bool)
        with np.load(io_path(second_output / "model_selection_folds.npz"), allow_pickle=False) as data:
            second_training = np.asarray(data["training_masks"], dtype=bool)
        self.assertEqual(first_metadata, second_metadata)
        np.testing.assert_array_equal(first_training, second_training)

    def test_run_build_cv_folds_processes_generation_manifest(self) -> None:
        gamma_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        experiment_a = self._write_experiment_root(gamma_matrix, t_steps=10)
        experiment_b = self._write_experiment_root(gamma_matrix, t_steps=10)
        manifest_path = self._write_generation_manifest([experiment_a, experiment_b])

        class FakePyMetis:
            @staticmethod
            def part_graph(nparts, adjacency=None, recursive=None, contiguous=None):
                return 4, [0, 1, 2, 3, 4]

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=FakePyMetis(),
        ):
            output_paths = cv_folds.run_build_cv_folds(
                manifest_path,
                num_folds=5,
            )

        self.assertEqual(len(output_paths), 2)
        for output_path in output_paths:
            self.assertTrue((output_path / "model_selection_folds.npz").exists())
            self.assertTrue((output_path / "bundle_metadata.yaml").exists())
            self.assertTrue((output_path / "outer_layer.npz").exists())
            self.assertTrue((output_path / "fold_summary.csv").exists())

    def test_min_time_block_sizes_for_various_k(self) -> None:
        """Test that _min_time_block_sizes_for_folds generates correct sizes for k=1..10."""
        # k=1: should be (1,)
        sizes = cv_folds._min_time_block_sizes_for_folds(1)
        self.assertEqual(sizes, (1,))

        # k=3: should be (1, 2, 2)
        sizes = cv_folds._min_time_block_sizes_for_folds(3)
        self.assertEqual(sizes, (1, 2, 2))
        self.assertEqual(sum(sizes), 5)

        # k=5: should be (1, 2, 2, 2, 2)
        sizes = cv_folds._min_time_block_sizes_for_folds(5)
        self.assertEqual(sizes, (1, 2, 2, 2, 2))
        self.assertEqual(sum(sizes), 9)

        # k=10: should be (1, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        sizes = cv_folds._min_time_block_sizes_for_folds(10)
        self.assertEqual(len(sizes), 10)
        self.assertEqual(sizes[0], 1)
        self.assertTrue(all(s == 2 for s in sizes[1:]))
        self.assertEqual(sum(sizes), 19)

    def test_build_cv_folds_with_k3(self) -> None:
        """Test build_cv_folds with k=3 folds."""
        gamma_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        experiment_root = self._write_experiment_root(
            gamma_matrix,
            include_node_index=True,
            include_time_index=True,
            t_steps=10,
        )

        class FakePyMetis:
            @staticmethod
            def part_graph(nparts, adjacency=None, recursive=None, contiguous=None):
                self.assertEqual(nparts, 3)
                self.assertEqual(len(adjacency), 5)
                return 25, [0, 1, 0, 2, 1]

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=FakePyMetis(),
        ):
            output_root = cv_folds._run_build_cv_folds_for_experiment(
                experiment_root,
                num_folds=3,
                seed=42,
                contiguous=False,
            )

        # Check output directory structure
        self.assertTrue((output_root / "model_selection_folds.npz").exists())
        self.assertTrue((output_root / "bundle_metadata.yaml").exists())

        # Load and verify model_selection_folds masks
        with np.load(output_root / "model_selection_folds.npz", allow_pickle=False) as data:
            training_masks = data["training_masks"]
        self.assertEqual(training_masks.shape[0], 3)  # 3 folds
        self.assertEqual(training_masks.shape[1], 10)  # 10 time steps
        self.assertEqual(training_masks.shape[2], 5)  # 5 vertices

        # Check metadata
        bundle_metadata = OmegaConf.to_container(
            OmegaConf.load(output_root / "bundle_metadata.yaml"),
            resolve=True,
        )
        self.assertEqual(bundle_metadata["num_folds"], 3)
        self.assertEqual(bundle_metadata["split_kind"], "train_cv")

    def test_get_num_folds_from_search_uses_search_override(self) -> None:
        """Test that _get_num_folds_from_search respects search-level num_folds."""
        search_with_k3 = {"name": "test", "num_folds": 3}
        self.assertEqual(cv_runner._get_num_folds_from_search(search_with_k3), 3)

        search_with_k10 = {"name": "test", "num_folds": 10}
        self.assertEqual(cv_runner._get_num_folds_from_search(search_with_k10), 10)

    def test_get_num_folds_from_search_defaults_to_default_num_folds(self) -> None:
        """Test that _get_num_folds_from_search defaults to DEFAULT_NUM_FOLDS."""
        search_without_override = {"name": "test"}
        self.assertEqual(
            cv_runner._get_num_folds_from_search(search_without_override),
            cv_runner.DEFAULT_NUM_FOLDS,
        )


class ValidationTestSplitArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "experiments" / f".tmp_validation_test_splits_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_experiment_root(
        self,
        gamma_matrix: np.ndarray,
        *,
        t_steps: int = 10,
        include_time_index: bool = True,
    ) -> Path:
        experiment_root = self.root / f"experiment_{uuid.uuid4().hex[:8]}"
        experiment_root.mkdir(parents=True, exist_ok=True)
        np.save(experiment_root / "gamma_matrix.npy", np.asarray(gamma_matrix, dtype=float))
        n_nodes = int(gamma_matrix.shape[0])
        np.savez(
            experiment_root / "panel_data.npz",
            x=np.zeros((int(t_steps), n_nodes), dtype=float),
            z=np.zeros((int(t_steps), n_nodes), dtype=float),
        )
        np.save(experiment_root / "x_0.npy", np.zeros(n_nodes, dtype=float))
        np.save(experiment_root / "z_0.npy", np.zeros(n_nodes, dtype=float))
        if include_time_index:
            with (experiment_root / "time_index.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["WeekStartDate", "WeekEndDate", "iso_week"],
                )
                writer.writeheader()
                for time_index in range(int(t_steps)):
                    writer.writerow(
                        {
                            "WeekStartDate": f"2021-01-{time_index + 1:02d}",
                            "WeekEndDate": f"2021-01-{time_index + 2:02d}",
                            "iso_week": str(time_index + 1),
                        }
                    )
        return experiment_root

    def _write_generation_manifest(self, rows: list[dict[str, object]]) -> Path:
        manifest_path = self.root / f"generation_manifest_{uuid.uuid4().hex[:8]}.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return manifest_path

    def _write_fit_manifest_rows(self, rows: list[dict[str, object]]) -> Path:
        manifest_path = self.root / f"fit_manifest_{uuid.uuid4().hex[:8]}.csv"
        write_csv_rows(manifest_path, rows)
        return manifest_path

    def _write_us_county_experiment(
        self,
        *,
        num_nodes: int = 100,
        t_steps: int = 10,
    ) -> tuple[Path, Path]:
        experiment_name = "test_exp_a"
        experiment_root = self.root / experiment_name
        x_0 = np.asarray([1 if index % 2 == 0 else -1 for index in range(num_nodes)], dtype=np.int8)
        z_0 = np.full(num_nodes, -1, dtype=np.int8)
        x = np.asarray(
            [
                [1 if (time_index + node_index) % 2 == 0 else -1 for node_index in range(num_nodes)]
                for time_index in range(t_steps)
            ],
            dtype=np.int8,
        )
        z = np.asarray(
            [
                [1 if node_index <= (time_index % num_nodes) else -1 for node_index in range(num_nodes)]
                for time_index in range(t_steps)
            ],
            dtype=np.int8,
        )
        node_table = pd.DataFrame(
            {
                "fips": [f"010{index:02d}" for index in range(1, num_nodes + 1)],
                "node_index": list(range(num_nodes)),
                "county": [f"county_{index}" for index in range(num_nodes)],
                "state_name": ["Alabama"] * num_nodes,
            }
        )
        time_index = pd.DataFrame(
            {
                "WeekStartDate": pd.date_range("2021-01-03", periods=t_steps + 1, freq="W-SUN"),
                "WeekEndDate": pd.date_range("2021-01-09", periods=t_steps + 1, freq="W-SAT"),
                "iso_year": [2021] * (t_steps + 1),
                "iso_week": list(range(1, t_steps + 2)),
                "model_index": list(range(t_steps + 1)),
            }
        )
        x_all = np.vstack([x_0[None, :], x])
        z_all = np.vstack([z_0[None, :], z])
        panel = pd.DataFrame(
            {
                "WeekEndDate": np.repeat(time_index["WeekEndDate"].to_numpy(), num_nodes),
                "fips": np.tile(node_table["fips"].to_numpy(), t_steps + 1),
                "Outcome_pm1": x_all.reshape(-1),
                "Intervention_pm1": z_all.reshape(-1),
            }
        )
        gamma = sparse.csr_matrix(
            np.asarray(
                [
                    [1.0 if abs(row_index - column_index) == 1 else 0.0 for column_index in range(num_nodes)]
                    for row_index in range(num_nodes)
                ],
                dtype=float,
            )
            / 2.0
        )
        config = create_us_county_config(
            n_nodes=num_nodes,
            t_steps=t_steps,
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
            "node_count": num_nodes,
            "time_steps": t_steps,
        }
        save_us_county_experiment(
            experiment_root,
            config,
            metadata,
            gamma,
            pd.DataFrame(
                {
                    "fips": node_table["fips"].iloc[:-1].to_list(),
                    "neighbor_fips": node_table["fips"].iloc[1:].to_list(),
                }
            ),
            panel,
            node_table,
            time_index,
            x,
            z,
            x_0,
            z_0,
        )
        manifest_path = self._write_generation_manifest(
            [
                {
                    "experiment_name": experiment_name,
                    "experiment_slug": experiment_name,
                    "descriptor": experiment_name,
                    "experiment_path": str(experiment_root.resolve()),
                    "intervention_source": "real_data",
                    "graph_source": "contiguity",
                    "N": num_nodes,
                    "T": t_steps,
                    "has_truth": False,
                }
            ]
        )
        return experiment_root, manifest_path

    @staticmethod
    def _deterministic_fake_pymetis():
        class FakePyMetis:
            @staticmethod
            def part_graph(nparts, adjacency=None, recursive=None, contiguous=None):
                n = len(adjacency or [])
                if n <= 0:
                    return 0, []
                return 0, [min((index * int(nparts)) // n, int(nparts) - 1) for index in range(n)]

        return FakePyMetis()

    def _build_nontrivial_test_train_cv_fixture(
        self,
        *,
        outer_num_folds: int = 3,
        inner_num_folds: int = 3,
        test_fold_id: int = 1,
        t_steps: int = 12,
        num_nodes: int = 72,
        edge_probability: float = 0.08,
    ) -> tuple[Path, Path]:
        spec_path = self.root / f"generation_spec_{uuid.uuid4().hex[:8]}.yaml"
        spec = {
            "base": {
                "experiment_root": f"{self.root.as_posix()}/generated",
                "manifest_path": f"{self.root.as_posix()}/generated/generation_manifest.csv",
                "dimensions": {"N": int(num_nodes), "T": int(t_steps)},
                "generation": {"gibbs_sweeps": 1, "seed": 7},
                "x0": {
                    "generator": "bernoulli",
                    "params": {"p": 0.5, "fixed_val": None},
                },
                "graph": {
                    "source": "generated",
                    "generator": "erdos_renyi",
                    "params": {"p": float(edge_probability)},
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
                    "field_mode": "random_low_rank",
                    "field_params": {},
                    "scalars": {"beta": 0.2, "xi": 0.1, "eta": 0.05},
                },
            },
            "experiments": [{"name": "exp_a"}],
        }
        OmegaConf.save(OmegaConf.create(spec), spec_path)
        generation_manifest = run_generation(spec_path, overwrite=True)
        experiment_root = Path(read_csv_manifest(generation_manifest)[0]["experiment_path"]).resolve()
        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=self._deterministic_fake_pymetis(),
        ):
            split_output_root = uscounty_splits.build_validation_test_splits_for_experiment(
                experiment_root,
                outer_num_folds=outer_num_folds,
                test_fold_id=test_fold_id,
                inner_num_folds=inner_num_folds,
                overwrite=True,
            )
        return experiment_root, split_output_root

    @staticmethod
    def _supported_fold_ids_from_masks(
        training_masks: np.ndarray,
        validation_masks: np.ndarray,
    ) -> tuple[int, ...]:
        supported_fold_ids: list[int] = []
        for fold_index in range(int(training_masks.shape[0])):
            if int(np.count_nonzero(training_masks[fold_index])) <= 0:
                continue
            if int(np.count_nonzero(validation_masks[fold_index])) <= 0:
                continue
            supported_fold_ids.append(int(fold_index + 1))
        return tuple(supported_fold_ids)

    def test_model_selection_folds_treat_pattern_change_as_temporal_transition(self) -> None:
        gamma_matrix = np.zeros((6, 6), dtype=float)
        experiment_root = self._write_experiment_root(gamma_matrix, t_steps=5)
        outer_active_mask = np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
        outer_separator_mask = np.zeros_like(outer_active_mask, dtype=bool)

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=self._deterministic_fake_pymetis(),
        ):
            model_selection = cv_folds.build_model_selection_folds(
                experiment_root,
                num_folds=2,
                outer_active_mask=outer_active_mask,
                outer_separator_mask=outer_separator_mask,
            )

        blanket_summary = model_selection["metadata"]["blanket_summary"]
        self.assertTrue(blanket_summary["blanket_validation_passed"])
        self.assertEqual(blanket_summary["temporal_violation_edge_count"], 0)
        self.assertEqual(model_selection["metadata"]["num_pattern_transitions"], 1)
        self.assertEqual(model_selection["metadata"]["pattern_transition_time_indices"], [1])

    def test_build_test_train_cv_bundle_handles_outer_driven_pattern_changes(self) -> None:
        spec_path = self.root / f"generation_spec_{uuid.uuid4().hex[:8]}.yaml"
        spec = {
            "base": {
                "experiment_root": f"{self.root.as_posix()}/generated",
                "manifest_path": f"{self.root.as_posix()}/generated/generation_manifest.csv",
                "dimensions": {"N": 500, "T": 50},
                "generation": {"gibbs_sweeps": 100, "seed": 42},
                "x0": {
                    "generator": "bernoulli",
                    "params": {"p": 0.5, "fixed_val": None},
                },
                "graph": {
                    "source": "generated",
                    "generator": "erdos_renyi",
                    "params": {"p": 0.01},
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
                    "generator": "low_rank_probability",
                    "params": {
                        "singular_values": [1.0, 0.7, 0.49],
                        "probability_amplitude": 0.5,
                    },
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
                    "field_mode": "random_low_rank",
                    "field_params": {
                        "singular_values": [2.0, 0.7, 0.49],
                        "target_rms_fraction": 0.4,
                    },
                    "scalars": {"beta": -0.3, "xi": 0.8, "eta": 0.3},
                },
            },
            "experiments": [
                {
                    "name": "confounding_strong",
                    "generation": {"seed": 42},
                    "intervention": {
                        "params": {"singular_values": [1.0, 0.7, 0.49]},
                    },
                    "truth": {
                        "field_mode": "confounded_low_rank",
                        "field_params": {
                            "singular_values": [1.0, 0.8, 0.6],
                            "target_rms_fraction": 0.75,
                        },
                    },
                }
            ],
        }
        OmegaConf.save(OmegaConf.create(spec), spec_path)
        generation_manifest = run_generation(spec_path, overwrite=True)
        experiment_root = Path(read_csv_manifest(generation_manifest)[0]["experiment_path"]).resolve()

        bundle = cv_folds.build_test_train_cv_bundle(
            experiment_root,
            outer_num_folds=6,
            test_fold_id=1,
            inner_num_folds=7,
        )

        outer_summary = bundle["metadata"]["outer_blanket_summary"]
        inner_summary = bundle["metadata"]["model_selection"]["blanket_summary"]
        pattern_transition_time_indices = bundle["metadata"]["model_selection"][
            "pattern_transition_time_indices"
        ]

        self.assertTrue(outer_summary["blanket_validation_passed"])
        self.assertTrue(inner_summary["blanket_validation_passed"])
        self.assertEqual(inner_summary["temporal_violation_edge_count"], 0)
        self.assertGreater(len(pattern_transition_time_indices), 0)
        self.assertIn(9, pattern_transition_time_indices)
        self.assertIn(10, pattern_transition_time_indices)

    def test_test_train_cv_outer_layer_masks_are_disjoint_and_nontrivial(self) -> None:
        experiment_root, split_output_root = self._build_nontrivial_test_train_cv_fixture()

        with np.load(io_path(split_output_root / "outer_layer.npz"), allow_pickle=False) as data:
            outer_active_mask = np.asarray(data["outer_active_mask"], dtype=bool)
            outer_separator_mask = np.asarray(data["outer_separator_mask"], dtype=bool)
            outer_test_mask = np.asarray(data["outer_test_mask"], dtype=bool)

        bundle_metadata = OmegaConf.to_container(
            OmegaConf.load(io_path(split_output_root / "bundle_metadata.yaml")),
            resolve=True,
        )

        self.assertEqual(bundle_metadata["split_kind"], "test_train_cv")
        self.assertEqual(outer_active_mask.shape, outer_separator_mask.shape)
        self.assertEqual(outer_active_mask.shape, outer_test_mask.shape)
        self.assertGreater(int(np.count_nonzero(outer_active_mask)), 0)
        self.assertGreater(int(np.count_nonzero(outer_separator_mask)), 0)
        self.assertGreater(int(np.count_nonzero(outer_test_mask)), 0)
        self.assertEqual(int(np.count_nonzero(outer_active_mask & outer_separator_mask)), 0)
        self.assertEqual(int(np.count_nonzero(outer_active_mask & outer_test_mask)), 0)
        self.assertEqual(int(np.count_nonzero(outer_separator_mask & outer_test_mask)), 0)

    def test_test_train_cv_inner_masks_respect_outer_active_and_outer_test_invariants(self) -> None:
        _, split_output_root = self._build_nontrivial_test_train_cv_fixture()

        with np.load(io_path(split_output_root / "outer_layer.npz"), allow_pickle=False) as data:
            outer_active_mask = np.asarray(data["outer_active_mask"], dtype=bool)
            outer_separator_mask = np.asarray(data["outer_separator_mask"], dtype=bool)
            outer_test_mask = np.asarray(data["outer_test_mask"], dtype=bool)
        with np.load(io_path(split_output_root / "model_selection_folds.npz"), allow_pickle=False) as data:
            training_masks = np.asarray(data["training_masks"], dtype=bool)
            separator_masks = np.asarray(data["separator_masks"], dtype=bool)
            validation_masks = np.asarray(data["validation_masks"], dtype=bool)

        outer_active_tensor = outer_active_mask[None, :, :]
        outer_separator_tensor = outer_separator_mask[None, :, :]
        outer_test_tensor = outer_test_mask[None, :, :]
        all_inner_masks = training_masks | separator_masks | validation_masks

        self.assertEqual(int(np.count_nonzero(training_masks & ~outer_active_tensor)), 0)
        self.assertEqual(int(np.count_nonzero(validation_masks & ~outer_active_tensor)), 0)
        self.assertEqual(int(np.count_nonzero(training_masks & outer_separator_tensor)), 0)
        self.assertEqual(int(np.count_nonzero(validation_masks & outer_separator_tensor)), 0)
        self.assertEqual(int(np.count_nonzero(training_masks & outer_test_tensor)), 0)
        self.assertEqual(int(np.count_nonzero(validation_masks & outer_test_tensor)), 0)
        self.assertEqual(int(np.count_nonzero(separator_masks & outer_test_tensor)), 0)
        self.assertEqual(
            int(np.count_nonzero(all_inner_masks & ~(outer_active_tensor | outer_separator_tensor))),
            0,
        )
        self.assertEqual(int(np.count_nonzero(training_masks & separator_masks)), 0)
        self.assertEqual(int(np.count_nonzero(training_masks & validation_masks)), 0)
        self.assertEqual(int(np.count_nonzero(separator_masks & validation_masks)), 0)

    def test_test_train_cv_supported_fold_ids_match_nonempty_inner_masks(self) -> None:
        _, split_output_root = self._build_nontrivial_test_train_cv_fixture()

        bundle_metadata = OmegaConf.to_container(
            OmegaConf.load(io_path(split_output_root / "bundle_metadata.yaml")),
            resolve=True,
        )
        with np.load(io_path(split_output_root / "model_selection_folds.npz"), allow_pickle=False) as data:
            training_masks = np.asarray(data["training_masks"], dtype=bool)
            validation_masks = np.asarray(data["validation_masks"], dtype=bool)

        expected_supported_fold_ids = self._supported_fold_ids_from_masks(
            training_masks,
            validation_masks,
        )
        self.assertEqual(
            tuple(int(value) for value in bundle_metadata["model_selection"]["supported_fold_ids"]),
            expected_supported_fold_ids,
        )

    def test_validation_test_split_artifacts_are_stable_for_repeated_runs(self) -> None:
        experiment_root, _ = self._build_nontrivial_test_train_cv_fixture()

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=self._deterministic_fake_pymetis(),
        ):
            first_output = uscounty_splits.build_validation_test_splits_for_experiment(
                experiment_root,
                outer_num_folds=3,
                inner_num_folds=3,
                overwrite=True,
            )
            outer_first = (first_output / "bundle_metadata.yaml").read_text(encoding="utf-8")
            with np.load(io_path(first_output / "model_selection_folds.npz"), allow_pickle=False) as data:
                first_training_masks = np.asarray(data["training_masks"], dtype=bool)

            second_output = uscounty_splits.build_validation_test_splits_for_experiment(
                experiment_root,
                outer_num_folds=3,
                inner_num_folds=3,
                overwrite=True,
            )
            outer_second = (second_output / "bundle_metadata.yaml").read_text(
                encoding="utf-8"
            )
            with np.load(io_path(second_output / "model_selection_folds.npz"), allow_pickle=False) as data:
                second_training_masks = np.asarray(data["training_masks"], dtype=bool)

        self.assertEqual(outer_first, outer_second)
        np.testing.assert_array_equal(first_training_masks, second_training_masks)

    def test_create_validation_test_splits_writes_us_county_artifacts_from_manifest(self) -> None:
        spec_path = self.root / f"generation_spec_{uuid.uuid4().hex[:8]}.yaml"
        spec = {
            "base": {
                "experiment_root": f"{self.root.as_posix()}/generated",
                "manifest_path": f"{self.root.as_posix()}/generated/generation_manifest.csv",
                "dimensions": {"N": 72, "T": 12},
                "generation": {"gibbs_sweeps": 1, "seed": 7},
                "x0": {
                    "generator": "bernoulli",
                    "params": {"p": 0.5, "fixed_val": None},
                },
                "graph": {
                    "source": "generated",
                    "generator": "erdos_renyi",
                    "params": {"p": 0.08},
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
                    "field_mode": "random_low_rank",
                    "field_params": {},
                    "scalars": {"beta": 0.2, "xi": 0.1, "eta": 0.05},
                },
            },
            "experiments": [{"name": "exp_a"}],
        }
        OmegaConf.save(OmegaConf.create(spec), spec_path)
        manifest_path = run_generation(spec_path, overwrite=True)

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=self._deterministic_fake_pymetis(),
        ):
            output_paths = uscounty_splits.create_validation_test_splits(
                manifest_path,
                outer_num_folds=3,
                inner_num_folds=3,
                overwrite=True,
            )

        self.assertEqual(len(output_paths), 1)
        output_root = output_paths[0]
        self.assertTrue(os.path.exists(io_path(output_root / "outer_layer.npz")))
        self.assertTrue(os.path.exists(io_path(output_root / "bundle_metadata.yaml")))
        self.assertTrue(os.path.exists(io_path(output_root / "model_selection_folds.npz")))
        self.assertTrue(os.path.exists(io_path(output_root / "bundle_metadata.yaml")))
        self.assertTrue(os.path.exists(io_path(output_root / "fold_summary.csv")))

    def test_load_outer_test_split_masks_matches_saved_npz(self) -> None:
        experiment_root, _ = self._write_us_county_experiment()

        split_output_root = uscounty_splits.build_validation_test_splits_for_experiment(
            experiment_root,
            outer_num_folds=2,
            inner_num_folds=2,
            overwrite=True,
        )

        loaded = load_outer_test_split_masks(
            experiment_root,
            test_fold_id=1,
            outer_num_folds=2,
            inner_num_folds=2,
        )
        with np.load(io_path(split_output_root / "outer_layer.npz"), allow_pickle=False) as data:
            np.testing.assert_array_equal(
                loaded["training_mask"],
                np.asarray(data["outer_active_mask"], dtype=bool),
            )
            np.testing.assert_array_equal(
                loaded["test_mask"],
                np.asarray(data["outer_test_mask"], dtype=bool),
            )
            np.testing.assert_array_equal(
                loaded["separator_mask"],
                np.asarray(data["outer_separator_mask"], dtype=bool),
            )

    def test_load_outer_training_split_masks_matches_saved_train_cv_npz(self) -> None:
        gamma_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        experiment_root = self._write_experiment_root(gamma_matrix, t_steps=10)
        manifest_path = self._write_generation_manifest(
            [
                {
                    "experiment_name": "exp_a",
                    "experiment_slug": "exp_a",
                    "experiment_path": str(experiment_root.resolve()),
                }
            ]
        )

        with mock.patch.object(
            cv_folds,
            "_load_pymetis",
            return_value=self._deterministic_fake_pymetis(),
        ):
            output_roots = cv_folds.run_build_cv_folds(
                manifest_path,
                num_folds=2,
                overwrite=True,
            )
        split_output_root = output_roots[0]

        loaded = load_outer_training_split_masks(
            experiment_root,
            split_kind="train_cv",
            num_folds=2,
        )
        with np.load(io_path(split_output_root / "outer_layer.npz"), allow_pickle=False) as data:
            np.testing.assert_array_equal(
                loaded["training_mask"],
                np.asarray(data["outer_active_mask"], dtype=bool),
            )
            np.testing.assert_array_equal(
                loaded["test_mask"],
                np.asarray(data["outer_test_mask"], dtype=bool),
            )
            np.testing.assert_array_equal(
                loaded["separator_mask"],
                np.asarray(data["outer_separator_mask"], dtype=bool),
            )

    def test_load_model_selection_split_masks_rejects_missing_inner_masks(self) -> None:
        experiment_root, _ = self._write_us_county_experiment()
        with self.assertRaises(FileNotFoundError):
            load_model_selection_split_masks(
                experiment_root,
                split_kind="test_train_cv",
                num_folds=5,
                outer_num_folds=5,
                test_fold_id=1,
            )

    def test_load_model_selection_split_masks_rejects_mismatched_fold_count(self) -> None:
        experiment_root, _ = self._write_us_county_experiment()

        split_output_root = uscounty_splits.build_validation_test_splits_for_experiment(
            experiment_root,
            outer_num_folds=2,
            inner_num_folds=2,
            overwrite=True,
        )

        with np.load(io_path(split_output_root / "model_selection_folds.npz"), allow_pickle=False) as data:
            np.savez(
                io_path(split_output_root / "model_selection_folds.npz"),
                training_masks=np.asarray(data["training_masks"], dtype=bool)[:-1],
                separator_masks=np.asarray(data["separator_masks"], dtype=bool)[:-1],
                validation_masks=np.asarray(data["validation_masks"], dtype=bool)[:-1],
            )
        with self.assertRaisesRegex(ValueError, "expected"):
            load_model_selection_split_masks(
                experiment_root,
                split_kind="test_train_cv",
                num_folds=2,
                outer_num_folds=2,
                test_fold_id=1,
            )

    def test_load_model_selection_split_masks_rejects_mismatched_spatial_shape(self) -> None:
        experiment_root, _ = self._write_us_county_experiment()

        split_output_root = uscounty_splits.build_validation_test_splits_for_experiment(
            experiment_root,
            outer_num_folds=2,
            inner_num_folds=2,
            overwrite=True,
        )

        with np.load(io_path(split_output_root / "model_selection_folds.npz"), allow_pickle=False) as data:
            training_masks = np.asarray(data["training_masks"], dtype=bool)
            np.savez(
                io_path(split_output_root / "model_selection_folds.npz"),
                training_masks=training_masks[:, :, :-1],
                separator_masks=np.asarray(data["separator_masks"], dtype=bool)[:, :, :-1],
                validation_masks=np.asarray(data["validation_masks"], dtype=bool)[:, :, :-1],
            )
        with self.assertRaisesRegex(ValueError, "expected"):
            load_model_selection_split_masks(
                experiment_root,
                split_kind="test_train_cv",
                num_folds=2,
                outer_num_folds=2,
                test_fold_id=1,
            )

    def test_create_validation_test_splits_rejects_manifest_row_without_experiment_path(self) -> None:
        manifest_path = self._write_generation_manifest(
            [{"experiment_name": "exp_a", "experiment_path": ""}]
        )
        with self.assertRaisesRegex(ValueError, "without experiment_path"):
            uscounty_splits.create_validation_test_splits(manifest_path)

    def test_create_validation_test_splits_rejects_invalid_test_fold_id(self) -> None:
        gamma_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        experiment_root = self._write_experiment_root(gamma_matrix, t_steps=10)
        with self.assertRaisesRegex(ValueError, "test_fold_id must be between 1 and 5"):
            uscounty_splits.build_validation_test_splits_for_experiment(
                experiment_root,
                test_fold_id=6,
            )

    def test_create_validation_test_splits_rejects_nonpositive_fold_counts(self) -> None:
        gamma_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        experiment_root = self._write_experiment_root(gamma_matrix, t_steps=10)
        with self.assertRaisesRegex(ValueError, "outer_num_folds must be >= 1"):
            uscounty_splits.build_validation_test_splits_for_experiment(
                experiment_root,
                outer_num_folds=0,
            )
        with self.assertRaisesRegex(ValueError, "inner_num_folds must be >= 1"):
            uscounty_splits.build_validation_test_splits_for_experiment(
                experiment_root,
                inner_num_folds=0,
            )

    def test_create_validation_test_splits_rejects_missing_gamma_artifact(self) -> None:
        experiment_root = self.root / "missing_gamma"
        experiment_root.mkdir(parents=True, exist_ok=True)
        np.savez(
            experiment_root / "panel_data.npz",
            x=np.zeros((10, 5), dtype=float),
            z=np.zeros((10, 5), dtype=float),
        )
        with self.assertRaisesRegex(FileNotFoundError, "Missing gamma matrix artifact"):
            uscounty_splits.build_validation_test_splits_for_experiment(experiment_root)

    def test_create_validation_test_splits_rejects_missing_time_and_panel_artifacts(self) -> None:
        gamma_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        experiment_root = self.root / "missing_panel"
        experiment_root.mkdir(parents=True, exist_ok=True)
        np.save(experiment_root / "gamma_matrix.npy", np.asarray(gamma_matrix, dtype=float))
        with self.assertRaisesRegex(FileNotFoundError, "panel_data.npz does not exist"):
            uscounty_splits.build_validation_test_splits_for_experiment(experiment_root)

    def test_create_validation_test_splits_rejects_too_short_time_horizon_for_outer_and_inner(self) -> None:
        gamma_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        short_experiment_root = self._write_experiment_root(gamma_matrix, t_steps=8)
        with self.assertRaisesRegex(ValueError, "at least 9 time steps"):
            uscounty_splits.build_validation_test_splits_for_experiment(
                short_experiment_root,
                outer_num_folds=5,
            )

        gamma_matrix_k6 = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        long_enough_for_outer = self._write_experiment_root(gamma_matrix_k6, t_steps=10)
        with self.assertRaisesRegex(ValueError, "at least 11 time steps"):
            uscounty_splits.build_validation_test_splits_for_experiment(
                long_enough_for_outer,
                outer_num_folds=5,
                inner_num_folds=6,
            )

    def test_run_test_evaluation_writes_default_report_with_expected_keys(self) -> None:
        experiment_root, _ = self._write_us_county_experiment()
        fit_root = experiment_root / "fits" / "bundle_a"
        fit_root.mkdir(parents=True, exist_ok=True)
        save_estimated_parameter_bundle(
            fit_root / "estimated_parameter_bundle.npz",
            beta=0.25,
            xi=0.1,
            eta=-0.2,
            latent_rank=0,
            t_steps=10,
            field_matrix=np.zeros((10, 100), dtype=float),
        )
        OmegaConf.save(
            OmegaConf.create(
                {
                    "estimation_params": {
                        "fixed_scalar_params": {},
                        "beta_mask_pre_s": True,
                        "beta_mask_post_e": False,
                    }
                }
            ),
            fit_root / "fit_realized_config.yaml",
        )
        OmegaConf.save(
            OmegaConf.create({"experiment_path": str(experiment_root.resolve())}),
            fit_root / "fit_metadata.yaml",
        )

        uscounty_splits.build_validation_test_splits_for_experiment(
            experiment_root,
            outer_num_folds=2,
            inner_num_folds=2,
            overwrite=True,
        )

        fit_manifest_path = self._write_fit_manifest_rows(
            [
                {
                    "fit_path": str(fit_root.resolve()),
                    "experiment_path": str(experiment_root.resolve()),
                    "split_kind": "test_train_cv",
                    "outer_num_folds": 2,
                    "test_fold_id": 1,
                    "num_folds": 2,
                }
            ]
        )
        results = run_test_evaluation(fit_manifest_path)
        self.assertEqual(results["num_evaluated_rows"], 1)
        self.assertEqual(results["num_skipped_rows"], 0)
        report_path = Path(results["evaluated_report_paths"][0])
        self.assertEqual(
            report_path,
            fit_root
            / "test_set_evaluation"
            / "test_fold_1__inner_folds_2"
            / "test_metrics.yaml",
        )
        payload = load_yaml_mapping(report_path)
        for key in (
            "training_loss",
            "num_training_slots",
            "test_loss",
            "test_brier_score",
            "test_ece",
            "num_test_slots",
            "post_s_test_loss",
            "post_s_test_brier_score",
            "post_s_test_ece",
            "num_post_s_test_slots",
            "test_mean_magnetization_abs_diff",
            "test_observed_mean_magnetization",
            "test_sampled_mean_magnetization_mean",
            "post_s_test_mean_magnetization_abs_diff",
            "post_s_test_observed_mean_magnetization",
            "post_s_test_sampled_mean_magnetization_mean",
            "full_panel_num_all_slots",
            "full_panel_all_mean_magnetization_abs_diff",
            "full_panel_num_test_slots",
            "full_panel_test_mean_magnetization_abs_diff",
            "full_panel_num_separator_slots",
            "full_panel_separator_mean_magnetization_abs_diff",
            "full_panel_num_treated_test_slots",
            "full_panel_treated_test_mean_magnetization_abs_diff",
            "full_panel_num_untreated_test_slots",
            "full_panel_untreated_test_mean_magnetization_abs_diff",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["experiment_path"], str(experiment_root.resolve()))

    def test_run_test_evaluation_infers_test_train_cv_settings_from_train_fit_metadata(self) -> None:
        experiment_root, _ = self._build_nontrivial_test_train_cv_fixture(
            outer_num_folds=3,
            inner_num_folds=3,
            test_fold_id=1,
            t_steps=12,
            num_nodes=72,
        )
        fit_root = (
            experiment_root
            / "train_fits"
            / "mask_grid"
            / "test_train_cv__outer_3__test_1__inner_3"
            / "candidate_a"
        )
        fit_root.mkdir(parents=True, exist_ok=True)
        save_estimated_parameter_bundle(
            fit_root / "estimated_parameter_bundle.npz",
            beta=0.2,
            xi=0.1,
            eta=0.05,
            latent_rank=0,
            t_steps=12,
            field_matrix=np.zeros((12, 72), dtype=float),
        )
        OmegaConf.save(
            OmegaConf.create(
                {
                    "estimation_params": {
                        "fixed_scalar_params": {},
                        "beta_mask_pre_s": False,
                        "beta_mask_post_e": False,
                    }
                }
            ),
            fit_root / "fit_realized_config.yaml",
        )
        OmegaConf.save(
            OmegaConf.create(
                {
                    "experiment_path": str(experiment_root.resolve()),
                    "execution_mode": "train_fit",
                    "split_kind": "test_train_cv",
                    "outer_num_folds": 3,
                    "test_fold_id": 1,
                    "num_folds": 3,
                }
            ),
            fit_root / "fit_metadata.yaml",
        )
        fit_manifest_path = self._write_fit_manifest_rows(
            [
                {
                    "fit_path": str(fit_root.resolve()),
                    "experiment_path": str(experiment_root.resolve()),
                    "split_kind": "test_train_cv",
                    "outer_num_folds": 3,
                    "test_fold_id": 1,
                    "num_folds": 3,
                }
            ]
        )
        results = run_test_evaluation(fit_manifest_path)
        self.assertEqual(results["num_evaluated_rows"], 1)
        report_path = Path(results["evaluated_report_paths"][0])

        self.assertEqual(
            report_path,
            fit_root
            / "test_set_evaluation"
            / "test_fold_1__inner_folds_3"
            / "test_metrics.yaml",
        )
        payload = load_yaml_mapping(report_path)
        self.assertEqual(payload["experiment_path"], str(experiment_root.resolve()))
        self.assertEqual(payload["split_kind"], "test_train_cv")
        self.assertEqual(int(payload["outer_num_folds"]), 3)
        self.assertEqual(int(payload["test_fold_id"]), 1)
        self.assertEqual(int(payload["inner_num_folds"]), 3)
        self.assertIn("test_loss", payload)
        self.assertIn("test_brier_score", payload)

    def test_run_test_evaluation_experiment_path_override_beats_stale_fit_metadata(self) -> None:
        experiment_root, _ = self._write_us_county_experiment()
        fit_root = experiment_root / "fits" / "bundle_b"
        fit_root.mkdir(parents=True, exist_ok=True)
        save_estimated_parameter_bundle(
            fit_root / "estimated_parameter_bundle.npz",
            beta=0.1,
            xi=0.05,
            eta=0.0,
            latent_rank=0,
            t_steps=10,
            field_matrix=np.zeros((10, 100), dtype=float),
        )
        OmegaConf.save(
            OmegaConf.create(
                {"estimation_params": {"fixed_scalar_params": {}, "beta_mask_pre_s": False}}
            ),
            fit_root / "fit_realized_config.yaml",
        )
        OmegaConf.save(
            OmegaConf.create({"experiment_path": str((self.root / "wrong_exp").resolve())}),
            fit_root / "fit_metadata.yaml",
        )

        uscounty_splits.build_validation_test_splits_for_experiment(
            experiment_root,
            outer_num_folds=2,
            inner_num_folds=2,
            overwrite=True,
        )

        fit_manifest_path = self._write_fit_manifest_rows(
            [
                {
                    "fit_path": str(fit_root.resolve()),
                    "experiment_path": str(experiment_root.resolve()),
                    "split_kind": "test_train_cv",
                    "outer_num_folds": 2,
                    "test_fold_id": 1,
                    "num_folds": 2,
                }
            ]
        )
        results = run_test_evaluation(fit_manifest_path)
        self.assertEqual(results["num_evaluated_rows"], 1)
        report_path = Path(results["evaluated_report_paths"][0])
        payload = load_yaml_mapping(report_path)
        self.assertEqual(payload["experiment_path"], str(experiment_root.resolve()))

    def test_run_test_evaluation_skips_train_cv_rows_in_mixed_manifest(self) -> None:
        experiment_root, _ = self._write_us_county_experiment()
        fit_root = experiment_root / "fits" / "bundle_skip"
        fit_root.mkdir(parents=True, exist_ok=True)
        save_estimated_parameter_bundle(
            fit_root / "estimated_parameter_bundle.npz",
            beta=0.1,
            xi=0.05,
            eta=0.0,
            latent_rank=0,
            t_steps=10,
            field_matrix=np.zeros((10, 100), dtype=float),
        )
        OmegaConf.save(
            OmegaConf.create(
                {"estimation_params": {"fixed_scalar_params": {}, "beta_mask_pre_s": False}}
            ),
            fit_root / "fit_realized_config.yaml",
        )
        OmegaConf.save(
            OmegaConf.create(
                {
                    "experiment_path": str(experiment_root.resolve()),
                    "execution_mode": "train_fit",
                    "split_kind": "train_cv",
                    "num_folds": 2,
                }
            ),
            fit_root / "fit_metadata.yaml",
        )
        fit_manifest_path = self._write_fit_manifest_rows(
            [
                {
                    "fit_path": str(fit_root.resolve()),
                    "experiment_path": str(experiment_root.resolve()),
                    "split_kind": "train_cv",
                    "num_folds": 2,
                }
            ]
        )

        results = run_test_evaluation(fit_manifest_path)

        self.assertEqual(results["num_evaluated_rows"], 0)
        self.assertEqual(results["num_skipped_rows"], 1)
        self.assertEqual(results["skipped_rows"][0]["split_kind"], "train_cv")


if __name__ == "__main__":
    unittest.main()
