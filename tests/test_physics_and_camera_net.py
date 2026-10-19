"""Mac checks for the pre-contact label and the camera network. No Isaac Sim."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from semantic_locomotion.front_friction_net import (
    MU_MAE_GATE,
    FrontFrictionNet,
    load_net,
    save_net,
)
from semantic_locomotion.physics_patches import (
    CROSSING_SPAWN_X,
    LOOKAHEAD_M,
    WET_BOTTOM_Z_M,
    WET_TOP_Z_M,
    WET_X,
    camera_ground_span_x,
    camera_pitch_rad,
    camera_ros_quat_wxyz,
    crossing_spawn_is_precontact,
    features_ahead,
    material_features,
    wet_slab_equilibrium_penetration_m,
)


class PhysicsPatchTests(unittest.TestCase):
    def test_lookahead_changes_before_the_base_enters_the_wet_slab(self):
        # Base is still on rock. The point the camera is aimed at is already wet.
        base_x = WET_X[0] - 0.5 * LOOKAHEAD_M
        self.assertLess(base_x, WET_X[0])
        root_xy = torch.tensor([[base_x, 0.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        ahead = features_ahead(root_xy, quat)
        under_base = material_features(root_xy)
        self.assertAlmostEqual(float(ahead[0, 0]), 0.15, places=5)
        self.assertAlmostEqual(float(under_base[0, 0]), 0.80, places=5)
        self.assertGreater(under_base[0, 1], ahead[0, 1])

    def test_features_match_the_two_physx_materials(self):
        rock = material_features(torch.tensor([[0.2, 0.0]]))
        wet = material_features(torch.tensor([[2.0, 0.0]]))
        self.assertAlmostEqual(float(rock[0, 0]), 0.80, places=5)
        self.assertAlmostEqual(float(wet[0, 0]), 0.15, places=5)

    def test_pitched_camera_centers_lookahead_point_on_ground(self):
        pitch = camera_pitch_rad()
        x_near, x_center, x_far = camera_ground_span_x(pitch)
        self.assertAlmostEqual(x_center, LOOKAHEAD_M, places=5)
        self.assertLess(x_near, LOOKAHEAD_M)
        self.assertGreater(x_far, LOOKAHEAD_M)
        quat = camera_ros_quat_wxyz(pitch)
        norm = sum(q * q for q in quat) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_crossing_spawn_sees_the_slab_before_the_base_enters_it(self):
        for spawn_x in CROSSING_SPAWN_X:
            self.assertTrue(crossing_spawn_is_precontact(spawn_x))
            root_xy = torch.tensor([[spawn_x, 0.0]])
            quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            ahead = features_ahead(root_xy, quat)
            under_base = material_features(root_xy)
            self.assertAlmostEqual(float(ahead[0, 0]), 0.15, places=5)
            self.assertAlmostEqual(float(under_base[0, 0]), 0.80, places=5)

    def test_wet_slab_thickness_prevents_rock_plane_bottom_out(self):
        # Top is 2 cm above the rock plane (z = 0.02 m) and extends down to z = -0.08 m.
        self.assertAlmostEqual(WET_TOP_Z_M, 0.02, places=5)
        self.assertLess(WET_BOTTOM_Z_M, 0.0)
        # Single diagonal-gait foot load (~74 N) and peak dynamic load (~120 N)
        # must compress less than the 10 cm compliant slab thickness.
        slab_thickness = WET_TOP_Z_M - WET_BOTTOM_Z_M
        self.assertLess(wet_slab_equilibrium_penetration_m(74.0), slab_thickness)
        self.assertLess(wet_slab_equilibrium_penetration_m(120.0), slab_thickness)


class FrontFrictionNetTests(unittest.TestCase):
    def test_rgb_batch_maps_to_two_numbers_and_reloads(self):
        net = FrontFrictionNet()
        rgb = torch.randint(0, 255, (4, 64, 64, 3), dtype=torch.uint8)
        out = net(rgb)
        self.assertEqual(tuple(out.shape), (4, 2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "front_friction.pt"
            save_net(net, path, val_mu_mae=0.02)
            restored = load_net(path)
        again = restored(rgb)
        self.assertTrue(torch.allclose(out, again))

    def test_failed_mae_gate_deletes_stale_checkpoint_and_blocks_load(self):
        net = FrontFrictionNet()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "front_friction.pt"
            save_net(net, path, val_mu_mae=0.03)
            self.assertTrue(path.is_file())
            with self.assertRaises(ValueError):
                save_net(net, path, val_mu_mae=MU_MAE_GATE + 0.05)
            self.assertFalse(path.is_file())
            with self.assertRaises(FileNotFoundError):
                load_net(path)

    def test_load_net_rejects_unverified_checkpoint_payload(self):
        net = FrontFrictionNet()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "front_friction.pt"
            torch.save({"state_dict": net.state_dict(), "gate_passed": False, "val_mu_mae": 0.25}, path)
            with self.assertRaises(RuntimeError):
                load_net(path)


if __name__ == "__main__":
    unittest.main()
