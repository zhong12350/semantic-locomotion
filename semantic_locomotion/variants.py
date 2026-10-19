"""Four evaluation modes. Oracle and Vision share the actor semantic shape."""

from __future__ import annotations

from dataclasses import dataclass

from semantic_locomotion.observations import PRIVILEGED_NAMES, SEMANTIC_DIM, SEMANTIC_NAMES


LOCO_TERMS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "height_scan",
)

# Actor must not see these, including in the Vision deployment.
FORBIDDEN_ACTOR_KEYS = ("material_id", "semantic_id", "friction", "mu", "gt_dry", "gt_wet")


@dataclass(frozen=True)
class Variant:
    name: str
    semantic_in_actor: bool
    semantic_source: str
    critic_privileged: bool


VARIANTS = {
    "baseline": Variant("baseline", False, "none", False),
    "oracle": Variant("oracle", True, "simulator", True),
    "vision": Variant("vision", True, "rgb_prediction", True),
    "shuffled": Variant("shuffled", True, "swapped_prediction", True),
}


def actor_contract(name: str) -> dict:
    spec = VARIANTS[name]
    semantic = list(SEMANTIC_NAMES) if spec.semantic_in_actor else []
    return {
        "variant": spec.name,
        "loco_terms": list(LOCO_TERMS),
        "semantic_terms": semantic,
        "semantic_dim": SEMANTIC_DIM if spec.semantic_in_actor else 0,
        "semantic_source": spec.semantic_source,
        "forbidden_actor_keys": list(FORBIDDEN_ACTOR_KEYS),
        "critic_privileged_terms": list(PRIVILEGED_NAMES) if spec.critic_privileged else [],
        "do_not_edit": "rsl_rl/algorithms/ppo.py",
    }
