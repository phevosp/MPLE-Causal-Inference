"""Small regression tests for the minimal active MPLE pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys
import shutil
import uuid

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.synthetic_data_generation import load_fixed_intervention_artifacts
from model_utils import (
    build_basis_expansion,
    compose_interaction_matrix,
    get_interaction_coeffs,
    interaction_features,
    validate_basis_infinity_norms,
)


def base_config() -> object:
    return OmegaConf.create(
        {
            "global_params": {
                "N": 4,
                "T": 3,
                "basis_params": {
                    "field_mode": "uniform",
                    "interaction_mode": "known_graph",
                    "num_shared_features": 2,
                    "shared_feature_seed": 0,
                },
            },
            "estimation_params": {
                "field_coefs": [],
                "interaction_coefs": [0.25],
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
        basis = build_basis_expansion(config, gamma)
        self.assertEqual(basis.interaction_names, ("adjacency",))
        validate_basis_infinity_norms(basis.field_basis, basis.interaction_basis)

    def test_interaction_coefficient_must_be_scalar(self) -> None:
        config = base_config()
        self.assertTrue(np.array_equal(get_interaction_coeffs(config), np.array([0.25])))
        with self.assertRaises(ValueError):
            compose_interaction_matrix(np.array([0.1, 0.2]), np.eye(4))

    def test_interaction_feature_shape_stays_single_template(self) -> None:
        x = np.array([[1, -1, 1, -1], [-1, -1, 1, 1]], dtype=float)
        gamma = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        features = interaction_features(x, gamma)
        self.assertEqual(features.shape, (1, 2, 4))

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


if __name__ == "__main__":
    unittest.main()
