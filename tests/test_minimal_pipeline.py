"""Small regression tests for the minimal active MPLE pipeline."""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
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
    build_synthetic_field,
    compose_interaction_matrix,
    compose_latent_field_matrix,
    get_xi,
    interaction_matrix_infinity_norm,
    interaction_effect,
    load_model_artifacts,
    load_true_parameters,
    save_model_artifacts,
    unpack_theta,
    validate_basis_infinity_norms,
)


def base_config() -> object:
    return OmegaConf.create(
        {
            "global_params": {
                "N": 4,
                "T": 3,
                "B": 1.0,
                "basis_params": {
                    "field_mode": "uniform",
                    "num_shared_features": 2,
                    "shared_feature_seed": 0,
                },
            },
            "estimation_params": {
                "field_coefs": [],
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
    def test_known_graph_basis_is_single_template(self) -> None:
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
        self.assertEqual(artifacts.field_mode, "uniform")
        validate_basis_infinity_norms(artifacts.field_basis, artifacts.gamma_matrix)
        self.assertEqual(artifacts.field_matrix.shape, (3, 4))
        self.assertIsNone(getattr(config.estimation_params, "tau_params", None))

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

    def test_latent_field_mode_is_realized(self) -> None:
        config = base_config()
        config.global_params.basis_params.field_mode = "latent_feature_matrix"
        config.global_params.basis_params.latent_rank = 2
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
            self.assertEqual(loaded.field_mode, "uniform")
            self.assertEqual(
                theta.shape[0], len(loaded.field_names) + config.global_params.T + 5
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_canonicalization_enforces_B_on_scalars_and_interaction(self) -> None:
        gamma = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float)
        artifacts = ModelArtifacts(
            field_mode="uniform",
            gamma_matrix=gamma,
            field_basis=np.empty((0, 2), dtype=float),
            field_names=(),
        )
        # theta layout for additive + fit_intervention_model=True with T=1:
        # [tau_0, beta, xi, eta, zeta, psi]
        theta = np.array([0.0, 5.0, 4.0, -3.5, 2.5, -2.2], dtype=float)
        projected = _canonicalize_theta(
            theta=theta,
            artifacts=artifacts,
            t_steps=1,
            fit_intervention_model=True,
            tau_zero_mean=False,
            bound_B=1.0,
        )
        parts = unpack_theta(projected, artifacts, t_steps=1, fit_intervention_model=True)
        self.assertLessEqual(abs(float(parts["beta"])), 1.0 + 1e-12)
        self.assertLessEqual(abs(float(parts["eta"])), 1.0 + 1e-12)
        self.assertLessEqual(abs(float(parts["zeta"])), 1.0 + 1e-12)
        self.assertLessEqual(abs(float(parts["psi"])), 1.0 + 1e-12)
        interaction = compose_interaction_matrix(float(parts["xi"]), artifacts.gamma_matrix)
        self.assertLessEqual(
            interaction_matrix_infinity_norm(interaction),
            1.0 + 1e-12,
        )


if __name__ == "__main__":
    unittest.main()
