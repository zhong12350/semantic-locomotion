#!/usr/bin/env python3
"""Supervise the front-camera CNN on Isaac Lab ground truth.

The robot is reset along the corridor so the camera sees rock and the wet slab.
Labels are the friction and stiffness of the ground 0.45 m ahead of the base.
Run this before scripts/train_semantic_go2.py.

    cd /workspace/isaaclab
    PYTHONPATH=/path/to/research/semantic-locomotion:$PYTHONPATH \\
    ./isaaclab.sh -p /path/to/research/semantic-locomotion/scripts/train_front_friction.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _isaaclab_root() -> Path:
    env = os.environ.get("ISAACLAB_PATH")
    if env:
        return Path(env)
    default = Path("/workspace/isaaclab")
    if (default / "isaaclab.sh").exists():
        return default
    raise SystemExit("Set ISAACLAB_PATH to the Isaac Lab v2.2.0 directory.")


def _launch():
    sys.path.insert(0, str(_isaaclab_root() / "scripts" / "reinforcement_learning" / "rsl_rl"))
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train the front-camera friction CNN.")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--val_rounds", type=int, default=16)
    parser.add_argument("--steps_per_round", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--max_mu_mae", type=float, default=None, help="Override MU_MAE_GATE.")
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    args.enable_cameras = True
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app
    return args, app


def _stratified_corridor_reset(env) -> None:
    """Half the envs look at rock ahead, half at wet (yaw = 0, local x fixed)."""
    import torch

    robot = env.scene["robot"]
    device = env.device
    n = env.num_envs
    origins = env.scene.env_origins
    half = n // 2
    env_ids = torch.arange(n, device=device)
    root = robot.data.root_pos_w.clone()
    root[:half, 0] = origins[:half, 0] + 0.25
    root[:half, 1] = origins[:half, 1]
    root[half:, 0] = origins[half:, 0] + 2.0
    root[half:, 1] = origins[half:, 1]
    root[:, 2] = origins[:, 2] + 0.40
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
    zero_vel = torch.zeros(n, 6, device=device)
    pose = torch.cat([root, quat], dim=-1)
    if hasattr(robot, "write_root_pose_to_sim"):
        robot.write_root_pose_to_sim(pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
    elif hasattr(robot, "write_root_state_to_sim"):
        state = torch.cat([root, quat, zero_vel], dim=-1)
        robot.write_root_state_to_sim(state, env_ids=env_ids)
    else:
        print("[WARN] stratified reset skipped: robot has no write_root_pose_to_sim", flush=True)


def _batch_stats(target) -> tuple[float, float]:
    import torch

    mu = target[:, 0]
    rock = (mu > 0.5).float().mean()
    wet = 1.0 - float(rock)
    return float(rock), float(wet)


def main() -> int:
    args_cli, simulation_app = _launch()

    import gymnasium as gym
    import torch
    import torch.nn.functional as F

    import isaaclab_tasks  # noqa: F401

    print(
        "[train_front_friction] v3.1 class-head + stratified reset + ground-crop + save-before-close",
        flush=True,
    )

    from semantic_locomotion.front_friction_net import (
        MU_MAE_GATE,
        FrontFrictionNet,
        checkpoint_path,
        save_net,
    )
    from semantic_locomotion.isaaclab_task.go2_semantic_cfg import UnitreeGo2SemanticEnvCfg, register
    from semantic_locomotion.isaaclab_task.observations import ground_truth_mu_k

    gate = MU_MAE_GATE if args_cli.max_mu_mae is None else float(args_cli.max_mu_mae)

    ckpt = checkpoint_path()
    if ckpt.is_file():
        ckpt.unlink()

    register()
    cfg = UnitreeGo2SemanticEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.precontact_mu_k = None

    env = gym.make("Isaac-Velocity-Semantic-Unitree-Go2-v0", cfg=cfg)
    device = env.unwrapped.device
    net = FrontFrictionNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args_cli.lr)
    zeros = torch.zeros((cfg.scene.num_envs, env.unwrapped.action_manager.total_action_dim), device=device)

    def _forward_loss(rgb, target) -> tuple[torch.Tensor, float, float]:
        logits = net.logits(rgb)
        labels = (target[:, 0] < 0.5).long()
        loss = F.cross_entropy(logits, labels)
        pred = net(rgb)
        mu_mae = float((pred.detach()[:, 0] - target[:, 0]).abs().mean())
        acc = float((logits.argmax(dim=-1) == labels).float().mean())
        return loss, mu_mae, acc

    for round_idx in range(args_cli.rounds):
        env.reset()
        _stratified_corridor_reset(env.unwrapped)
        env.step(zeros)
        rgb = env.unwrapped.scene.sensors["front_camera"].data.output["rgb"]
        target = ground_truth_mu_k(env.unwrapped)
        rock_frac, wet_frac = _batch_stats(target)
        mu_mae_last = 0.0
        loss_last = 0.0
        acc_last = 0.0
        for _ in range(max(1, args_cli.steps_per_round)):
            opt.zero_grad(set_to_none=True)
            loss, mu_mae_last, acc_last = _forward_loss(rgb, target)
            loss.backward()
            opt.step()
            loss_last = float(loss.detach())
        print(
            f"round {round_idx:03d}  ce {loss_last:.5f}  mu_mae {mu_mae_last:.4f}  acc {acc_last:.3f}"
            f"  rock_frac {rock_frac:.2f}  wet_frac {wet_frac:.2f}",
            flush=True,
        )

    net.eval()
    val_errors: list[float] = []
    with torch.no_grad():
        for _ in range(max(1, args_cli.val_rounds)):
            env.reset()
            _stratified_corridor_reset(env.unwrapped)
            env.step(zeros)
            rgb = env.unwrapped.scene.sensors["front_camera"].data.output["rgb"]
            target = ground_truth_mu_k(env.unwrapped)
            pred = net(rgb)
            val_errors.append(float((pred[:, 0] - target[:, 0]).abs().mean()))
    val_mu_mae = sum(val_errors) / len(val_errors)

    print(f"[GATE] val_mu_mae={val_mu_mae:.4f}  threshold={gate:.4f}", flush=True)
    try:
        saved = save_net(net, ckpt, val_mu_mae=val_mu_mae, max_mu_mae=gate)
    except ValueError as exc:
        print(
            f"{exc}\nCamera network did not pass the gate. Do not start locomotion training.",
            file=sys.stderr,
            flush=True,
        )
        try:
            env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass
        return 1

    print(f"saved {saved}  val_mu_mae {val_mu_mae:.4f}", flush=True)
    # Save before tearing down Isaac Sim; close() has been observed to abort the process
    # after training metrics already look good, leaving an empty checkpoints/ directory.
    try:
        env.close()
    except Exception as exc:
        print(f"[WARN] env.close() raised: {exc}", flush=True)
    try:
        simulation_app.close()
    except Exception as exc:
        print(f"[WARN] simulation_app.close() raised: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
