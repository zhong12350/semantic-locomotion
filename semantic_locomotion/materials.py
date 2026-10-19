"""Two floor materials for the first cave demo.

Geometry stays similar. Appearance and friction differ.
"""

from __future__ import annotations

from dataclasses import dataclass


DRY_ROCK = "dry_rock"
WET_SLIPPERY = "wet_slippery"
CLASS_ORDER = (DRY_ROCK, WET_SLIPPERY)


@dataclass(frozen=True)
class Material:
    name: str
    mu: float
    # RGB in 0-1, used only as a debug swatch until Isaac materials exist.
    albedo_rgb: tuple[float, float, float]


MATERIALS: dict[str, Material] = {
    DRY_ROCK: Material(DRY_ROCK, mu=0.80, albedo_rgb=(0.45, 0.38, 0.30)),
    WET_SLIPPERY: Material(WET_SLIPPERY, mu=0.15, albedo_rgb=(0.12, 0.16, 0.18)),
}


def material(name: str) -> Material:
    try:
        return MATERIALS[name]
    except KeyError as exc:
        raise ValueError(f"unknown material {name!r}, expected {CLASS_ORDER}") from exc
