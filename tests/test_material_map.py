from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_locomotion.materials import MATERIALS
from semantic_locomotion.terrain import eval_layout, material_at, mu_at, slip_proxy, upcoming


class MaterialMapTests(unittest.TestCase):
    def test_dry_and_wet_differ_in_friction_and_color(self):
        dry = MATERIALS["dry_rock"]
        wet = MATERIALS["wet_slippery"]
        self.assertGreater(dry.mu, wet.mu)
        self.assertNotEqual(dry.albedo_rgb, wet.albedo_rgb)

    def test_eval_layout_is_flat_and_traversable(self):
        terrain = eval_layout()
        self.assertTrue(terrain.same_height)
        self.assertEqual(material_at(terrain, np.array([1.0, 0.0])), "dry_rock")
        self.assertEqual(material_at(terrain, np.array([5.0, 0.0])), "wet_slippery")
        self.assertEqual(material_at(terrain, np.array([9.0, 0.0])), "dry_rock")
        self.assertGreater(mu_at(terrain, np.array([1.0, 0.0])), mu_at(terrain, np.array([5.0, 0.0])))

    def test_same_command_slips_more_on_wet(self):
        terrain = eval_layout()
        dry = slip_proxy(mu_at(terrain, np.array([1.0, 0.0])))
        wet = slip_proxy(mu_at(terrain, np.array([5.0, 0.0])))
        self.assertEqual(dry, 0.0)
        self.assertGreater(wet, dry)

    def test_upcoming_class_changes_before_the_foot_crosses(self):
        terrain = eval_layout()
        foot = np.array([3.2, 0.0])
        self.assertEqual(material_at(terrain, foot), "dry_rock")
        query = upcoming(terrain, foot, yaw=0.0, lookahead_m=1.5)
        self.assertEqual(query.upcoming, "wet_slippery")
        self.assertLess(query.distance_m, 1.5)
        self.assertTrue(query.transition_inside_horizon)
        contacted = foot + np.array([query.distance_m, 0.0])
        self.assertEqual(material_at(terrain, contacted), "wet_slippery")
