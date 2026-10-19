"""Semantic locomotion package.

The Isaac Lab task in ``isaaclab_task`` is what trains. It subclasses the official
Go2 flat task and appends a 2-D pre-contact vector, ``(mu, log10(K) / 5)``.
``rsl_rl/algorithms/ppo.py`` is not modified.

``materials``, ``terrain``, ``observations``, and ``variants`` are the Mac-side
4-D contract (``p_dry, p_wet, confidence, transition_distance``). The simulator
does not import them.
"""

from semantic_locomotion.materials import MATERIALS, DRY_ROCK, WET_SLIPPERY
from semantic_locomotion.observations import SEMANTIC_DIM, SEMANTIC_NAMES

__all__ = [
    "MATERIALS",
    "DRY_ROCK",
    "WET_SLIPPERY",
    "SEMANTIC_DIM",
    "SEMANTIC_NAMES",
]
