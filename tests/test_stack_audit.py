from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_locomotion.stack_audit import audit_local


class StackAuditTests(unittest.TestCase):
    def test_local_rsl_rl_is_mlp_only(self):
        data = audit_local()
        self.assertEqual(data["rsl_rl_version"], "2.3.1")
        self.assertFalse(data["has_cnn_encoder"])
        self.assertFalse(data["gpu_host_audited"])
        self.assertFalse(data["edit_ppo"])
        self.assertIn("ActorCritic", data["policy_classes"])
        self.assertNotIn("ActorCriticCNN", data["policy_classes"])
        self.assertIsNone(data["checkpoint_actor_obs"])
        self.assertIn("critic", data["critic_observation"])
