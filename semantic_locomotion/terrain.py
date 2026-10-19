"""Cave-corridor material map.

Height is constant. Full-width patches change only along x, so a forward
camera and a foot can disagree before contact. This is a spatial contract,
not a PhysX scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_locomotion.materials import CLASS_ORDER, MATERIALS, material


@dataclass(frozen=True)
class Patch:
    x0: float
    x1: float
    name: str


@dataclass(frozen=True)
class Upcoming:
    current: str
    upcoming: str
    distance_m: float
    transition_inside_horizon: bool


@dataclass(frozen=True)
class CaveCorridor:
    length_m: float
    width_m: float
    patches: tuple[Patch, ...]
    seed: int

    def __post_init__(self) -> None:
        if self.length_m <= 0 or self.width_m <= 0:
            raise ValueError("corridor size must be positive")
        cursor = 0.0
        for patch in self.patches:
            if patch.name not in MATERIALS:
                raise ValueError(f"unknown material {patch.name!r}")
            if patch.x0 != cursor or patch.x1 <= patch.x0:
                raise ValueError("patches must tile [0, length] without gaps or overlaps")
            cursor = patch.x1
        if not np.isclose(cursor, self.length_m):
            raise ValueError("patches must cover the full corridor length")

    @property
    def same_height(self) -> bool:
        return True


def eval_layout(*, seed: int = 0) -> CaveCorridor:
    """Fixed dry-wet-dry route used for paired evaluation."""
    return CaveCorridor(
        length_m=12.0,
        width_m=2.0,
        patches=(
            Patch(0.0, 4.0, "dry_rock"),
            Patch(4.0, 8.0, "wet_slippery"),
            Patch(8.0, 12.0, "dry_rock"),
        ),
        seed=seed,
    )


def inside(terrain: CaveCorridor, xy: np.ndarray) -> bool:
    xy = np.asarray(xy, dtype=np.float64)
    half = 0.5 * terrain.width_m
    return bool(0.0 <= xy[0] < terrain.length_m and -half <= xy[1] <= half)


def material_at(terrain: CaveCorridor, xy: np.ndarray) -> str:
    xy = np.asarray(xy, dtype=np.float64)
    if not inside(terrain, xy):
        raise ValueError(f"query {xy.tolist()} is outside the corridor")
    for patch in terrain.patches:
        if patch.x0 <= xy[0] < patch.x1:
            return patch.name
    raise RuntimeError("patch tiling left a hole")


def mu_at(terrain: CaveCorridor, xy: np.ndarray) -> float:
    return material(material_at(terrain, xy)).mu


def upcoming(
    terrain: CaveCorridor,
    xy: np.ndarray,
    yaw: float,
    *,
    lookahead_m: float = 1.5,
    step_m: float = 0.01,
) -> Upcoming:
    """Walk along the heading and return the first material change."""
    if lookahead_m <= 0 or step_m <= 0:
        raise ValueError("lookahead_m and step_m must be positive")
    origin = np.asarray(xy, dtype=np.float64)
    current = material_at(terrain, origin)
    direction = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    n = int(np.floor(lookahead_m / step_m))
    for i in range(1, n + 1):
        point = origin + direction * (i * step_m)
        if not inside(terrain, point):
            break
        name = material_at(terrain, point)
        if name != current:
            return Upcoming(current, name, i * step_m, True)
    return Upcoming(current, current, lookahead_m, False)


def slip_proxy(mu: float, *, command_speed: float = 1.0, mu_no_slip: float = 0.8) -> float:
    """Ordering check only. PhysX must still be measured on the GPU host.

    Same commanded speed, lower friction, larger proxy. Zero at ``mu_no_slip``.
    """
    if command_speed < 0 or mu < 0 or mu_no_slip <= 0:
        raise ValueError("slip proxy arguments must be non-negative, mu_no_slip > 0")
    return float(command_speed * max(0.0, 1.0 - mu / mu_no_slip))


def class_index(name: str) -> int:
    return CLASS_ORDER.index(material(name).name)
