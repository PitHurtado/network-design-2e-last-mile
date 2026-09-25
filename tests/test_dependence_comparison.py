"""Small deterministic tests for the paired dependence comparison."""

import unittest

import numpy as np
import pandas as pd

from src.scenarios.generation.dependence import Independent, SpatialJoint
from src.scenarios.generation.generator import ScenarioGenerator
from src.scenarios.spatial import pixel_neighbor_pairs


class DependenceComparisonTests(unittest.TestCase):
    def test_modes_preserve_pairing_and_marginal_scale(self):
        n_pixels = 6
        cholesky = np.linalg.cholesky(np.full((n_pixels, n_pixels), 0.25) + np.eye(n_pixels) * 0.75)
        kwargs = dict(
            pixels=[f"A-{i}" for i in range(n_pixels)],
            expected_stop=np.full((n_pixels, 12), 20.0),
            expected_drop=np.full((n_pixels, 12), 5.0),
            sigma_stop=np.full(n_pixels, 0.2),
            sigma_drop=np.full(n_pixels, 0.2),
            sigma_common_stop=0.0,
            sigma_common_drop=0.0,
        )
        seed = np.random.SeedSequence([123, 100]).spawn(1)[0]
        independent = ScenarioGenerator(**kwargs, dependence=Independent())
        joint = ScenarioGenerator(**kwargs, dependence=SpatialJoint(cholesky))
        independent_stop, independent_drop = independent.draw(np.random.default_rng(seed), 1.0)
        joint_stop, joint_drop = joint.draw(np.random.default_rng(seed), 1.0)

        self.assertEqual(independent_stop.shape, (n_pixels, 12))
        self.assertEqual(independent_drop.shape, (n_pixels, 12))
        self.assertTrue(np.all(independent_stop >= 1))
        self.assertTrue(np.all(independent_drop > 0))
        self.assertTrue(np.any(independent_drop != joint_drop))
        self.assertTrue(np.any(independent_stop != joint_stop))

    def test_neighbor_pairs_use_edge_adjacency(self):
        # Two pixels sharing an edge, one diagonal pixel, and one isolated pixel.
        crosswalk = pd.DataFrame(
            {
                "layer": ["A", "A", "A", "A"],
                "pixel": [1, 2, 3, 4],
                "cell": [0, 1, 17, 100],
            }
        )
        pairs = pixel_neighbor_pairs(
            pixels=["A-1", "A-2", "A-3", "A-4"],
            crosswalk=crosswalk,
            ring=1,
        )
        actual = {tuple(row) for row in pairs[["id_pixel", "neighbor"]].to_numpy()}
        self.assertIn(("A-1", "A-2"), actual)
        self.assertNotIn(("A-1", "A-3"), actual)
        self.assertNotIn(("A-1", "A-4"), actual)


if __name__ == "__main__":
    unittest.main()
