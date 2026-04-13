"""Small regression tests for the minimal latent-only MPLE pipeline."""

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


if __name__ == "__main__":
    unittest.main()
