"""Expand an rsl-rl 2.3.1 MLP checkpoint by appending semantic inputs.

Original first-layer columns are copied. New columns are zero, so a zero
semantic vector reproduces the old actor and critic. Later layers are copied
unchanged. Unknown layouts are refused.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


ACTOR_WEIGHT = "actor.0.weight"
CRITIC_WEIGHT = "critic.0.weight"


def _as_state_dict(checkpoint: Mapping) -> tuple[dict, dict]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if "model_state_dict" in checkpoint:
        raw = checkpoint["model_state_dict"]
        if not isinstance(raw, Mapping):
            raise ValueError("model_state_dict is not a mapping")
        return dict(checkpoint), dict(raw)
    return {}, dict(checkpoint)


def _require_linear(state: Mapping, key: str) -> torch.Tensor:
    if key not in state:
        raise ValueError(f"refusing checkpoint without {key}")
    weight = state[key]
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(f"{key} must be a rank-2 tensor, got {type(weight).__name__}")
    return weight


def _expand(weight: torch.Tensor, bias: torch.Tensor, extra: int) -> tuple[torch.Tensor, torch.Tensor]:
    if extra < 0:
        raise ValueError("extra input dimension must be >= 0")
    if bias.shape != (weight.shape[0],):
        raise ValueError(f"bias shape {tuple(bias.shape)} does not match weight {tuple(weight.shape)}")
    if extra == 0:
        return weight.clone(), bias.clone()
    expanded = weight.new_zeros(weight.shape[0], weight.shape[1] + extra)
    expanded[:, : weight.shape[1]] = weight
    return expanded, bias.clone()


def migrate_state_dict(
    checkpoint: Mapping,
    *,
    actor_extra: int,
    critic_extra: int,
) -> dict:
    """Return a new checkpoint dict. The input is not modified."""
    outer, state = _as_state_dict(checkpoint)
    actor_w = _require_linear(state, ACTOR_WEIGHT)
    critic_w = _require_linear(state, CRITIC_WEIGHT)
    if ACTOR_WEIGHT.replace("weight", "bias") not in state or CRITIC_WEIGHT.replace("weight", "bias") not in state:
        raise ValueError("refusing checkpoint without actor.0.bias and critic.0.bias")
    actor_b = state["actor.0.bias"]
    critic_b = state["critic.0.bias"]
    new_state = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in state.items()}
    new_state[ACTOR_WEIGHT], new_state["actor.0.bias"] = _expand(actor_w, actor_b, actor_extra)
    new_state[CRITIC_WEIGHT], new_state["critic.0.bias"] = _expand(critic_w, critic_b, critic_extra)
    if outer:
        migrated = dict(outer)
        migrated["model_state_dict"] = new_state
        migrated["semantic_migration"] = {
            "actor_extra": actor_extra,
            "critic_extra": critic_extra,
            "new_columns": "zero",
            "appended_at": "end",
        }
        return migrated
    return new_state


def zero_semantic_matches(
    old: nn.Module,
    migrated: nn.Module,
    old_obs: torch.Tensor,
    old_critic_obs: torch.Tensor,
    *,
    atol: float = 1e-5,
) -> bool:
    """True when appending zeros leaves actor and critic outputs unchanged."""
    actor_extra = migrated.actor[0].in_features - old.actor[0].in_features
    critic_extra = migrated.critic[0].in_features - old.critic[0].in_features
    if actor_extra < 0 or critic_extra < 0:
        raise ValueError("migrated model is smaller than the original")
    zeros_a = old_obs.new_zeros(old_obs.shape[0], actor_extra)
    zeros_c = old_critic_obs.new_zeros(old_critic_obs.shape[0], critic_extra)
    with torch.no_grad():
        old_action = old.act_inference(old_obs)
        new_action = migrated.act_inference(torch.cat([old_obs, zeros_a], dim=-1))
        old_value = old.evaluate(old_critic_obs)
        new_value = migrated.evaluate(torch.cat([old_critic_obs, zeros_c], dim=-1))
    return bool(torch.allclose(old_action, new_action, atol=atol) and torch.allclose(old_value, new_value, atol=atol))
