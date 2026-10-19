from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_locomotion.observations import (
    SEMANTIC_DIM,
    actor_semantic,
    mirror_longitudinal,
    missing_vector,
    oracle_vector,
    semantic_vector,
    shuffle_vector,
    vision_vector,
)
from semantic_locomotion.terrain import eval_layout
from semantic_locomotion.variants import actor_contract


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.terrain = eval_layout()
        self.xy = np.array([3.2, 0.0])

    def test_vector_is_normalized_and_clipped(self):
        vector = semantic_vector(2.0, 2.0, 3.0, 9.0, lookahead_m=1.5)
        self.assertEqual(vector.shape, (SEMANTIC_DIM,))
        np.testing.assert_allclose(vector, [0.5, 0.5, 1.0, 1.0])
        missing = semantic_vector(0.0, 0.0, 1.0, 0.2)
        np.testing.assert_allclose(missing, missing_vector() * 0 + np.array([0.5, 0.5, 0.0, 0.2 / 1.5]))

    def test_oracle_changes_before_contact(self):
        before = oracle_vector(self.terrain, np.array([2.4, 0.0]), 0.0)
        near = oracle_vector(self.terrain, self.xy, 0.0)
        np.testing.assert_allclose(before[:2], [1.0, 0.0])
        np.testing.assert_allclose(near[:2], [0.0, 1.0])
        self.assertEqual(near[2], 1.0)
        self.assertLess(near[3], 1.0)

    def test_oracle_vision_and_shuffled_share_shape(self):
        oracle = actor_semantic("oracle", self.terrain, self.xy, 0.0)
        vision = actor_semantic(
            "vision",
            self.terrain,
            self.xy,
            0.0,
            probabilities=np.array([0.1, 0.9]),
            confidence=0.7,
        )
        shuffled = actor_semantic(
            "shuffled",
            self.terrain,
            self.xy,
            0.0,
            probabilities=np.array([0.1, 0.9]),
            confidence=0.7,
        )
        self.assertIsNone(actor_semantic("baseline", self.terrain, self.xy, 0.0))
        self.assertEqual(oracle.shape, vision.shape)
        self.assertEqual(vision.shape, shuffled.shape)
        np.testing.assert_allclose(shuffled, shuffle_vector(vision))
        np.testing.assert_allclose(mirror_longitudinal(vision), vision)

    def test_missing_prediction_and_contracts(self):
        missing = actor_semantic("vision", self.terrain, self.xy, 0.0)
        np.testing.assert_allclose(missing, missing_vector())
        direct = vision_vector(np.array([0.2, 0.8]), -0.4, 0.3)
        self.assertEqual(direct[2], 0.0)
        oracle = actor_contract("oracle")
        vision = actor_contract("vision")
        self.assertEqual(oracle["semantic_dim"], vision["semantic_dim"])
        self.assertEqual(actor_contract("baseline")["semantic_dim"], 0)
        self.assertNotIn("mu", vision["semantic_terms"])
        self.assertIn("mu", vision["forbidden_actor_keys"])
