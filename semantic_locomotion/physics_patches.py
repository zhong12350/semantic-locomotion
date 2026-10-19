"""Two ground materials and camera-ground geometry in the env-local frame.

The rock is the terrain plane. The wet slab sits ahead of the spawn, so a
point 0.45 m in front of the base can already be on the wet material while
the base is still on rock. That is the pre-contact label, and the front camera
is pitched down so its optical axis hits that exact ground point at nominal
standing height.
"""

from __future__ import annotations

import math

import torch

LOOKAHEAD_M = 0.45

# Locomotion spawn sits just in front of the wet slab and faces +x.
# At every x in CROSSING_SPAWN_X the base is still on rock, while the
# 0.45 m lookahead point is already inside the slab.
CROSSING_SPAWN_X = (0.95, 1.10)
CROSSING_SPAWN_Y = (-0.30, 0.30)
CROSSING_SPAWN_YAW = (-0.25, 0.25)
CROSSING_LIN_VEL_X = (0.40, 0.80)
CROSSING_LIN_VEL_Y = (-0.10, 0.10)
CROSSING_HEADING = (-0.25, 0.25)

# Nominal base height and camera mount offset in the robot base frame.
NOMINAL_BASE_Z_M = 0.40
CAMERA_OFFSET_POS = (0.32, 0.0, 0.06)
CAMERA_FOCAL_LENGTH_MM = 24.0
CAMERA_HORIZONTAL_APERTURE_MM = 20.955

# Env-local AABB of the wet slab. Matches the cuboid spawned in the Isaac task.
WET_X = (1.4, 4.6)
WET_Y = (-2.0, 2.0)

# Slab thickness and pose.
# A 2 cm height step is not achievable when compliant_contact_stiffness = 2e3 N/m
# over a rock plane because a single diagonal-gait foot (~74 N) compresses the
# spring by ~3.7 cm and bottoms out onto the underlying rock plane.
# Instead, the slab is 0.10 m thick (10 cm spring clearance >> 3.7 cm) and its
# top sits 2 cm above the rock plane (top z = 0.02 m, center z = -0.03 m) so
# feet on the wet slab contact only the wet compliant material.
WET_SLAB_SIZE = (3.2, 4.0, 0.10)
WET_SLAB_POS = (3.0, 0.0, -0.03)
WET_TOP_Z_M = WET_SLAB_POS[2] + 0.5 * WET_SLAB_SIZE[2]
WET_BOTTOM_Z_M = WET_SLAB_POS[2] - 0.5 * WET_SLAB_SIZE[2]

MU_ROCK = 0.80
MU_WET = 0.15
K_ROCK = 2.0e5
K_WET = 2.0e3
LOG_K_SCALE = 5.0


def camera_pitch_rad(
    lookahead_m: float = LOOKAHEAD_M,
    base_z_m: float = NOMINAL_BASE_Z_M,
    cam_pos: tuple[float, float, float] = CAMERA_OFFSET_POS,
) -> float:
    """Downward pitch angle (rad) so the optical axis hits ``lookahead_m`` ahead on z = 0."""
    dx = lookahead_m - cam_pos[0]
    dz = base_z_m + cam_pos[2]
    if dx <= 0.0 or dz <= 0.0:
        raise ValueError(f"camera must be behind and above the lookahead point, got dx={dx}, dz={dz}")
    return math.atan2(dz, dx)


def camera_fov_half_angle_rad(
    focal_length_mm: float = CAMERA_FOCAL_LENGTH_MM,
    aperture_mm: float = CAMERA_HORIZONTAL_APERTURE_MM,
) -> float:
    return math.atan(0.5 * aperture_mm / focal_length_mm)


def camera_ros_quat_wxyz(pitch_rad: float | None = None) -> tuple[float, float, float, float]:
    """Quaternion ``(w, x, y, z)`` in Isaac Lab ``convention="ros"`` for a downward pitch.

    At ``pitch_rad = 0`` this equals ``(0.5, -0.5, 0.5, -0.5)`` (+Z forward along
    base +X, -Y up along base +Z). For ``pitch_rad > 0``, the camera pitches down
    around its +X (right) axis so the optical axis aims at the ground ahead.
    """
    if pitch_rad is None:
        pitch_rad = camera_pitch_rad()
    half = 0.5 * pitch_rad
    c = math.cos(half)
    s = math.sin(half)
    w0, x0, y0, z0 = 0.5, -0.5, 0.5, -0.5
    # Post-multiply by Rx(-pitch_rad) = (cos(half), -sin(half), 0, 0) in camera ROS frame.
    w = w0 * c + x0 * s
    x = x0 * c - w0 * s
    y = y0 * c + z0 * s
    z = z0 * c - y0 * s
    return (w, x, y, z)


def camera_ground_span_x(
    pitch_rad: float | None = None,
    base_z_m: float = NOMINAL_BASE_Z_M,
    cam_pos: tuple[float, float, float] = CAMERA_OFFSET_POS,
) -> tuple[float, float, float]:
    """Return ``(x_near, x_center, x_far)`` on z = 0 intersected by the vertical FOV."""
    if pitch_rad is None:
        pitch_rad = camera_pitch_rad()
    half_fov = camera_fov_half_angle_rad()
    cam_z = base_z_m + cam_pos[2]
    upper_depr = pitch_rad - half_fov
    lower_depr = pitch_rad + half_fov
    if upper_depr <= 0.0 or lower_depr >= math.pi:
        raise ValueError("vertical FOV rays do not point toward the ground plane")
    x_near = cam_pos[0] + cam_z * (math.cos(lower_depr) / math.sin(lower_depr))
    x_center = cam_pos[0] + cam_z * (math.cos(pitch_rad) / math.sin(pitch_rad))
    x_far = cam_pos[0] + cam_z * (math.cos(upper_depr) / math.sin(upper_depr))
    return (x_near, x_center, x_far)


def wet_slab_equilibrium_penetration_m(normal_force_n: float, stiffness: float = K_WET) -> float:
    """Static spring compression ``F / K`` on the wet slab before touching z = 0."""
    if normal_force_n < 0.0 or stiffness <= 0.0:
        raise ValueError("normal_force_n must be >= 0 and stiffness > 0")
    return normal_force_n / stiffness


def _yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def lookahead_xy(root_xy: torch.Tensor, yaw: torch.Tensor, lookahead_m: float = LOOKAHEAD_M) -> torch.Tensor:
    """Point on the ground the front camera is aimed at, in the env-local frame."""
    ahead = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1) * lookahead_m
    return root_xy + ahead


def is_on_wet_slab(xy: torch.Tensor) -> torch.Tensor:
    """Boolean mask for points inside the env-local AABB of the wet slab."""
    return (
        (xy[..., 0] >= WET_X[0])
        & (xy[..., 0] < WET_X[1])
        & (xy[..., 1] >= WET_Y[0])
        & (xy[..., 1] < WET_Y[1])
    )


def material_features(xy: torch.Tensor) -> torch.Tensor:
    """Return (mu, log10(K) / 5) for each point. Both components are O(1)."""
    wet = is_on_wet_slab(xy)
    mu = torch.where(wet, torch.full_like(xy[:, 0], MU_WET), torch.full_like(xy[:, 0], MU_ROCK))
    log_k = torch.where(
        wet,
        torch.full_like(xy[:, 0], math.log10(K_WET) / LOG_K_SCALE),
        torch.full_like(xy[:, 0], math.log10(K_ROCK) / LOG_K_SCALE),
    )
    return torch.stack((mu, log_k), dim=-1)


def crossing_spawn_is_precontact(spawn_x: float, lookahead_m: float = LOOKAHEAD_M) -> bool:
    """True when the base is still on rock and the lookahead point is on the wet slab."""
    if lookahead_m <= 0.0:
        raise ValueError("lookahead_m must be positive")
    return spawn_x < WET_X[0] <= spawn_x + lookahead_m


def features_ahead(root_xy: torch.Tensor, root_quat_wxyz: torch.Tensor) -> torch.Tensor:
    yaw = _yaw_wxyz(root_quat_wxyz)
    return material_features(lookahead_xy(root_xy, yaw))

