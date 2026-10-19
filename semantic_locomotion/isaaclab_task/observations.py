"""Observation terms that run only after Isaac Lab has created the scene.

``camera_mu_k`` is what the locomotion MLP receives. ``ground_truth_mu_k`` is
the same two numbers read from the slab the lookahead point is on, for the critic.
"""

from __future__ import annotations

import torch

from semantic_locomotion.front_friction_net import FrontFrictionNet, load_net
from semantic_locomotion.physics_patches import (
    WET_TOP_Z_M,
    features_ahead,
    is_on_wet_slab,
)

_NET: FrontFrictionNet | None = None
_STATE_ATTR = "_semantic_contact_state"
_FIRST_WET_STANCE_STEPS = 10


class _ContactState:
    """Per-env contact bookkeeping. A new scene gets a new instance."""

    def __init__(self) -> None:
        self.foot_body_ids = None
        self.contact_foot_ids = None
        self.slip_prev = None
        self.slip_origin = None
        self.slip_age = None
        self.slip_active = None
        self.slip_sum = None
        self.slip_count = None
        self.last_step: int | None = None
        self.last_metrics: dict[str, torch.Tensor] | None = None


def _contact_state(env) -> _ContactState:
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        state = _ContactState()
        setattr(env, _STATE_ATTR, state)
    return state


def reset_runtime_state(env) -> None:
    """Clear slip totals on this env before an evaluation rollout."""
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        return
    state.slip_prev = None
    state.slip_origin = None
    state.slip_age = None
    state.slip_active = None
    state.slip_sum = None
    state.slip_count = None
    state.last_step = None
    state.last_metrics = None


def _root_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    root_xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    quat = robot.data.root_quat_w
    return root_xy, quat


def log_foot_material_contacts(env) -> dict[str, torch.Tensor]:
    """Classify active foot contacts into wet-slab vs rock-plane and record in ``env.extras["log"]``.

    The actor and the critic can both call this in one step. The slip accumulator runs once.

    A foot with contact force > 1 N inside the wet slab AABB is counted as a pure
    wet-slab contact when its world Z stays above ``z = 0`` (the rock plane); if
    ``z <= 0`` inside the AABB, it is flagged as bottoming out onto the rock plane.
    """
    state = _contact_state(env)
    step_counter = getattr(env, "common_step_counter", None)
    if (
        step_counter is not None
        and state.last_metrics is not None
        and state.last_step == int(step_counter)
    ):
        env.extras.setdefault("log", {}).update(state.last_metrics)
        return state.last_metrics

    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]
    if state.foot_body_ids is None:
        state.foot_body_ids, _ = robot.find_bodies(".*_foot")
    if state.contact_foot_ids is None:
        state.contact_foot_ids, _ = contact_sensor.find_bodies(".*_foot")

    foot_pos_w = robot.data.body_pos_w[:, state.foot_body_ids, :]
    foot_xy_local = foot_pos_w[:, :, :2] - env.scene.env_origins[:, None, :2]
    foot_z = foot_pos_w[:, :, 2]

    forces = contact_sensor.data.net_forces_w[:, state.contact_foot_ids, :]
    in_contact = torch.norm(forces, dim=-1) > 1.0
    over_wet = is_on_wet_slab(foot_xy_local)

    wet_contact = in_contact & over_wet & (foot_z > 0.0)
    wet_bottom_out = in_contact & over_wet & (foot_z <= 0.0)
    rock_contact = in_contact & (~over_wet)
    slip_mean, slip_count = _update_first_wet_slip(env, wet_contact, foot_xy_local)

    metrics = {
        "Contact/wet_foot_fraction": wet_contact.float().mean(),
        "Contact/rock_foot_fraction": rock_contact.float().mean(),
        "Contact/wet_bottom_out_fraction": wet_bottom_out.float().mean(),
        "Contact/wet_surface_clearance_m": (foot_z - 0.0).clamp(max=WET_TOP_Z_M).mean(),
        "Contact/first_wet_slip_m": slip_mean,
        "Contact/first_wet_entries": slip_count,
    }
    log_dict = env.extras.setdefault("log", {})
    log_dict.update(metrics)
    if step_counter is not None:
        state.last_step = int(step_counter)
        state.last_metrics = metrics
    return metrics


def _update_first_wet_slip(env, wet_contact: torch.Tensor, foot_xy: torch.Tensor):
    """Horizontal travel during the first 0.2 s after a foot steps onto the wet slab."""
    state = _contact_state(env)
    n, feet, _ = foot_xy.shape
    device = foot_xy.device
    if state.slip_prev is None or state.slip_prev.shape != wet_contact.shape:
        state.slip_prev = torch.zeros_like(wet_contact)
        state.slip_origin = torch.zeros(n, feet, 2, device=device)
        state.slip_age = torch.zeros(n, feet, dtype=torch.long, device=device)
        state.slip_active = torch.zeros_like(wet_contact)
        state.slip_sum = torch.zeros((), device=device)
        state.slip_count = torch.zeros((), device=device)

    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None and bool(torch.any(reset_buf)):
        ids = reset_buf.nonzero(as_tuple=False).flatten()
        state.slip_prev[ids] = False
        state.slip_active[ids] = False
        state.slip_age[ids] = 0

    entered = wet_contact & ~state.slip_prev & ~state.slip_active
    state.slip_origin = torch.where(entered.unsqueeze(-1), foot_xy, state.slip_origin)
    state.slip_active = state.slip_active | entered
    state.slip_age = torch.where(state.slip_active, state.slip_age + 1, torch.zeros_like(state.slip_age))
    finished = state.slip_active & (state.slip_age >= _FIRST_WET_STANCE_STEPS)
    if bool(torch.any(finished)):
        disp = torch.linalg.norm(foot_xy - state.slip_origin, dim=-1)
        state.slip_sum = state.slip_sum + disp[finished].sum()
        state.slip_count = state.slip_count + finished.sum()
        state.slip_active = state.slip_active & ~finished
        state.slip_age = torch.where(finished, torch.zeros_like(state.slip_age), state.slip_age)
    state.slip_prev = wet_contact
    count = state.slip_count.detach()
    mean = state.slip_sum.detach() / count.clamp(min=1)
    return mean, count


def ground_truth_mu_k(env) -> torch.Tensor:
    root_xy, quat = _root_pose(env)
    log_foot_material_contacts(env)
    return features_ahead(root_xy, quat)


def camera_mu_k(env) -> torch.Tensor:
    global _NET
    if _NET is None:
        _NET = load_net(device=env.device)
    camera = env.scene.sensors["front_camera"]
    rgb = camera.data.output["rgb"]
    with torch.no_grad():
        return _NET(rgb)
