#!/usr/bin/env python3
"""Train blind, oracle, and camera Go2 policies in one Isaac session.

The camera CNN is already trained. This script does not retrain it. One launch
walks the same wet slab three times:

    blind   actor sees no pre-contact material
    oracle  actor sees the true (mu, log K) 0.45 m ahead
    camera  actor sees the frozen CNN estimate of that vector

Run on the Isaac Lab machine:

    cd /workspace/isaaclab
    PYTHONPATH=/workspace/semantic-locomotion:$PYTHONPATH \\
    ./isaaclab.sh -p /workspace/semantic-locomotion/scripts/train_semantic_go2.py \\
        --headless --num_envs 512

At 512 envs this is about 90 minutes. The comparison is written to
semantic-locomotion/suite_results/summary.json. It does not edit rsl_rl ppo.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONDITIONS = ("blind", "oracle", "camera")
EVAL_STEPS = 1000


def _camera_checkpoint() -> Path:
    return ROOT / "checkpoints" / "front_friction.pt"


def _isaaclab_root() -> Path:
    env = os.environ.get("ISAACLAB_PATH")
    if env:
        return Path(env)
    default = Path("/workspace/isaaclab")
    if (default / "isaaclab.sh").exists():
        return default
    raise SystemExit("Set ISAACLAB_PATH to the Isaac Lab v2.2.0 directory.")


def _launch():
    rsl_rl_scripts = _isaaclab_root() / "scripts" / "reinforcement_learning" / "rsl_rl"
    sys.path.insert(0, str(rsl_rl_scripts))
    from isaaclab.app import AppLauncher
    import cli_args

    parser = argparse.ArgumentParser(description="Train blind, oracle, and camera Go2 on one wet slab.")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iterations", type=int, default=1500)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--video_interval", type=int, default=2000)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument(
        "--condition",
        choices=[*CONDITIONS, "all"],
        default="all",
        help="all trains blind, then oracle, then camera in this process.",
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    args.enable_cameras = args.condition in ("camera", "all")
    args.task = "Isaac-Velocity-Semantic-Unitree-Go2-v0"
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app
    return args, app, cli_args


def _as_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _evaluate(env, runner, device: str) -> dict[str, float]:
    from semantic_locomotion.isaaclab_task.observations import reset_runtime_state

    base = env.unwrapped if hasattr(env, "unwrapped") else env
    reset_runtime_state(base)
    policy = runner.get_inference_policy(device=device)
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    averaged = (
        "Contact/wet_foot_fraction",
        "Metrics/base_velocity/error_vel_xy",
        "Metrics/base_velocity/error_vel_yaw",
    )
    latest = ("Contact/first_wet_slip_m", "Contact/first_wet_entries")
    totals = {key: 0.0 for key in averaged}
    counts = {key: 0 for key in averaged}
    last = {}
    for _ in range(EVAL_STEPS):
        actions = policy(obs)
        stepped = env.step(actions)
        obs = stepped[0]
        extras = stepped[-1] if isinstance(stepped[-1], dict) else {}
        log = extras.get("log", {})
        for key in averaged:
            if key in log:
                totals[key] += _as_float(log[key])
                counts[key] += 1
        for key in latest:
            if key in log:
                last[key] = _as_float(log[key])
    metrics = {key: totals[key] / counts[key] for key in totals if counts[key]}
    metrics.update(last)
    return metrics


def main() -> None:
    condition_arg = "all"
    if "--condition" in sys.argv:
        condition_arg = sys.argv[sys.argv.index("--condition") + 1]
    ckpt = _camera_checkpoint()
    needs_camera = condition_arg in ("camera", "all")
    if needs_camera and not ckpt.is_file():
        raise SystemExit(
            f"Missing {ckpt}. Run scripts/train_front_friction.py on this machine and pass the MAE gate first."
        )

    # AppLauncher must start before torch/gymnasium — importing them earlier breaks Isaac Sim numpy.
    args_cli, simulation_app, cli_args = _launch()

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    if needs_camera:
        from semantic_locomotion.front_friction_net import load_net

        try:
            load_net(ckpt, device="cpu")
        except RuntimeError as exc:
            raise SystemExit(f"{exc} Re-run train_front_friction.py and pass the MAE gate.") from exc

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_pickle, dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    import isaaclab_tasks  # noqa: F401

    from semantic_locomotion.isaaclab_task.go2_semantic_cfg import (
        UnitreeGo2SemanticEnvCfg,
        UnitreeGo2SemanticPPORunnerCfg,
        configure_wet_crossing,
        register,
    )

    register()
    torch.backends.cudnn.benchmark = False
    conditions = CONDITIONS if args_cli.condition == "all" else (args_cli.condition,)
    if args_cli.condition == "all" and args_cli.resume:
        raise SystemExit("Resume one condition with --condition blind|oracle|camera. Do not resume inside --condition all.")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_dir = ROOT / "suite_results" / stamp
    summary_dir.mkdir(parents=True, exist_ok=True)
    minutes = len(conditions) * args_cli.max_iterations * 1.2 / 60.0
    print(
        f"[SUITE] conditions={','.join(conditions)}  iters={args_cli.max_iterations}"
        f"  envs={args_cli.num_envs}  budget≈{minutes:.0f} min",
        flush=True,
    )

    def _train_one(condition: str) -> dict:
        env_cfg = UnitreeGo2SemanticEnvCfg()
        agent_cfg = UnitreeGo2SemanticPPORunnerCfg()
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg.max_iterations = args_cli.max_iterations
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        configure_wet_crossing(env_cfg, condition=condition)
        agent_cfg.experiment_name = f"unitree_go2_cross_{condition}"
        agent_cfg.run_name = condition
        print(
            f"[TASK] {condition}"
            f"  spawn_x={env_cfg.events.reset_base.params['pose_range']['x']}"
            f"  lin_vel_x={env_cfg.commands.base_velocity.ranges.lin_vel_x}",
            flush=True,
        )

        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root, log_dir)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        if args_cli.video and args_cli.condition != "all":
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.path.join(log_dir, "videos", "train"),
                step_trigger=lambda step: step % args_cli.video_interval == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )
            print_dict({"video_folder": os.path.join(log_dir, "videos", "train")}, nesting=4)

        resume_path = None
        if agent_cfg.resume:
            resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        if resume_path is not None:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        wrapped.reset()
        metrics = _evaluate(wrapped, runner, agent_cfg.device)
        env.close()
        metrics["log_dir"] = log_dir
        print(f"[SUITE] {condition} eval {metrics}", flush=True)
        return metrics

    summary: dict[str, dict] = {}
    for condition in conditions:
        print(f"[SUITE] start {condition}", flush=True)
        try:
            summary[condition] = _train_one(condition)
        except Exception as exc:
            summary[condition] = {"error": str(exc)}
            print(f"[SUITE] {condition} failed: {exc}", flush=True)
        _write_summary(summary_dir, summary)

    _print_summary(summary)
    print(f"[SUITE] wrote {summary_dir / 'summary.json'}", flush=True)
    simulation_app.close()


def _write_summary(summary_dir: Path, summary: dict) -> None:
    payload = {}
    for condition, row in summary.items():
        payload[condition] = {
            key: (value if isinstance(value, str) else float(value))
            for key, value in row.items()
        }
    (summary_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def _print_summary(summary: dict) -> None:
    print("[SUITE] comparison", flush=True)
    for condition in CONDITIONS:
        row = summary.get(condition)
        if row is None:
            continue
        if "error" in row:
            print(f"  {condition:8}  FAILED  {row['error']}", flush=True)
            continue
        wet = row.get("Contact/wet_foot_fraction", float("nan"))
        slip = row.get("Contact/first_wet_slip_m", float("nan"))
        vel = row.get("Metrics/base_velocity/error_vel_xy", float("nan"))
        print(f"  {condition:8}  wet_foot={wet:.4f}  first_slip_m={slip:.4f}  vel_xy_err={vel:.4f}", flush=True)


if __name__ == "__main__":
    main()
