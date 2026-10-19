from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "official" / "rsl_rl"))
# ActorCritic imports rsl_rl.utils, which imports GitPython at module level.
# The MLP itself does not use git. Stub it so the Mac test can load the real class.
sys.modules.setdefault("git", types.ModuleType("git"))

from rsl_rl.modules import ActorCritic

from semantic_locomotion.checkpoint import migrate_state_dict, zero_semantic_matches
from semantic_locomotion.observations import SEMANTIC_DIM


def _policy(actor_obs: int, critic_obs: int) -> ActorCritic:
    return ActorCritic(
        actor_obs,
        critic_obs,
        12,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        init_noise_std=0.1,
    )


class CheckpointMigrationTests(unittest.TestCase):
    def test_zero_semantic_matches_and_loads_strictly(self):
        old = _policy(20, 24)
        wrapped = {"model_state_dict": old.state_dict(), "iter": 3}
        original = wrapped["model_state_dict"]["actor.0.weight"].clone()
        migrated = migrate_state_dict(wrapped, actor_extra=SEMANTIC_DIM, critic_extra=SEMANTIC_DIM)
        self.assertTrue(torch.equal(wrapped["model_state_dict"]["actor.0.weight"], original))
        new = _policy(20 + SEMANTIC_DIM, 24 + SEMANTIC_DIM)
        self.assertTrue(new.load_state_dict(migrated["model_state_dict"], strict=True))
        obs = torch.randn(4, 20)
        critic_obs = torch.randn(4, 24)
        self.assertTrue(zero_semantic_matches(old, new, obs, critic_obs))
        with torch.no_grad():
            new.actor[0].weight[:, 20:] = 0.2
            same = new.act_inference(torch.cat([obs, torch.zeros(4, SEMANTIC_DIM)], dim=-1))
            changed = new.act_inference(torch.cat([obs, torch.ones(4, SEMANTIC_DIM)], dim=-1))
            self.assertTrue(torch.allclose(same, old.act_inference(obs), atol=1e-5))
            self.assertFalse(torch.allclose(changed, same, atol=1e-5))

    def test_refuses_unknown_layout(self):
        old = _policy(8, 8)
        state = old.state_dict()
        del state["actor.0.weight"]
        with self.assertRaises(ValueError):
            migrate_state_dict(state, actor_extra=4, critic_extra=4)
        bad = _policy(8, 8).state_dict()
        bad["actor.0.weight"] = torch.zeros(32)
        with self.assertRaises(ValueError):
            migrate_state_dict(bad, actor_extra=4, critic_extra=4)
        with self.assertRaises(ValueError):
            migrate_state_dict(old.state_dict(), actor_extra=-1, critic_extra=4)
