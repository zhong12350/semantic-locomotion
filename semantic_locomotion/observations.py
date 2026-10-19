"""Compact semantic vector shared by Oracle and Vision.

Layout: ``[p_dry, p_wet, confidence, transition_distance]``.
Distance is divided by the lookahead, so every component is in ``[0, 1]``.
Baseline omits the vector. The deployed actor never receives material id or mu.
"""

from __future__ import annotations

import numpy as np

from semantic_locomotion.materials import CLASS_ORDER, DRY_ROCK, WET_SLIPPERY, material
from semantic_locomotion.terrain import CaveCorridor, upcoming


SEMANTIC_NAMES = ("p_dry", "p_wet", "confidence", "transition_distance")
SEMANTIC_DIM = len(SEMANTIC_NAMES)
LOOKAHEAD_M = 1.5
MODES = ("baseline", "oracle", "vision", "shuffled")
# Critic-only fields. Vision actor input must not include these.
PRIVILEGED_NAMES = ("gt_dry", "gt_wet", "mu", "transition_distance")


def semantic_vector(
    p_dry: float,
    p_wet: float,
    confidence: float,
    distance_m: float,
    *,
    lookahead_m: float = LOOKAHEAD_M,
) -> np.ndarray:
    if lookahead_m <= 0:
        raise ValueError("lookahead_m must be positive")
    probs = np.array([p_dry, p_wet], dtype=np.float64)
    if np.any(probs < 0):
        raise ValueError("class probabilities must be non-negative")
    total = float(probs.sum())
    if total == 0.0:
        probs[:] = 0.5
        confidence = 0.0
    else:
        probs /= total
    conf = float(np.clip(confidence, 0.0, 1.0))
    dist = float(np.clip(distance_m, 0.0, lookahead_m) / lookahead_m)
    return np.array([probs[0], probs[1], conf, dist], dtype=np.float64)


def missing_vector(*, lookahead_m: float = LOOKAHEAD_M) -> np.ndarray:
    """No perception yet: uniform class, zero confidence, horizon distance."""
    return semantic_vector(0.5, 0.5, 0.0, lookahead_m, lookahead_m=lookahead_m)


def shuffle_vector(vector: np.ndarray) -> np.ndarray:
    """Swap the two class probabilities. Confidence and distance stay."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (SEMANTIC_DIM,):
        raise ValueError(f"expected shape {(SEMANTIC_DIM,)}, got {vector.shape}")
    return np.array([vector[1], vector[0], vector[2], vector[3]], dtype=np.float64)


def mirror_longitudinal(vector: np.ndarray) -> np.ndarray:
    """Left-right mirror of a full-width patch does not change the vector.

    Valid only while every patch spans the corridor width. Lateral patches
    would need a different rule.
    """
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (SEMANTIC_DIM,):
        raise ValueError(f"expected shape {(SEMANTIC_DIM,)}, got {vector.shape}")
    return vector.copy()


def oracle_vector(
    terrain: CaveCorridor,
    xy: np.ndarray,
    yaw: float,
    *,
    lookahead_m: float = LOOKAHEAD_M,
) -> np.ndarray:
    query = upcoming(terrain, xy, yaw, lookahead_m=lookahead_m)
    p_dry = 1.0 if query.upcoming == DRY_ROCK else 0.0
    p_wet = 1.0 if query.upcoming == WET_SLIPPERY else 0.0
    return semantic_vector(p_dry, p_wet, 1.0, query.distance_m, lookahead_m=lookahead_m)


def vision_vector(
    probabilities: np.ndarray,
    confidence: float,
    distance_m: float,
    *,
    lookahead_m: float = LOOKAHEAD_M,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (2,):
        raise ValueError("vision probabilities must have shape (2,) in CLASS_ORDER")
    return semantic_vector(
        float(probabilities[0]),
        float(probabilities[1]),
        confidence,
        distance_m,
        lookahead_m=lookahead_m,
    )


def privileged_vector(
    terrain: CaveCorridor,
    xy: np.ndarray,
    yaw: float,
    *,
    lookahead_m: float = LOOKAHEAD_M,
) -> np.ndarray:
    """Critic-only ground truth. Do not concatenate this onto the actor."""
    query = upcoming(terrain, xy, yaw, lookahead_m=lookahead_m)
    one_hot = np.zeros(len(CLASS_ORDER), dtype=np.float64)
    one_hot[CLASS_ORDER.index(query.upcoming)] = 1.0
    mu = material(query.upcoming).mu
    dist = float(np.clip(query.distance_m, 0.0, lookahead_m) / lookahead_m)
    return np.concatenate([one_hot, np.array([mu, dist], dtype=np.float64)])


def actor_semantic(
    mode: str,
    terrain: CaveCorridor,
    xy: np.ndarray,
    yaw: float,
    *,
    probabilities: np.ndarray | None = None,
    confidence: float | None = None,
    lookahead_m: float = LOOKAHEAD_M,
) -> np.ndarray | None:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "baseline":
        return None
    oracle = oracle_vector(terrain, xy, yaw, lookahead_m=lookahead_m)
    if mode == "oracle":
        return oracle
    if probabilities is None or confidence is None:
        return missing_vector(lookahead_m=lookahead_m)
    predicted = vision_vector(probabilities, confidence, oracle[3] * lookahead_m, lookahead_m=lookahead_m)
    if mode == "vision":
        return predicted
    return shuffle_vector(predicted)
