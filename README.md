# Visual semantic locomotion

The runnable task is a single Go2 on the official Isaac Lab flat velocity task. A frozen camera network appends `(mu, log10(K) / 5)` to the policy observation. The policy stays the LocoLab `ActorCritic` MLP. `rsl_rl/algorithms/ppo.py` is not modified.

The package root (`materials`, `terrain`, `observations`, `variants`) is a separate Mac-side contract for a 4-D vector `[p_dry, p_wet, confidence, transition_distance]`. The Isaac Lab task does not import it. Cave terrain and a second robot are later stages; they are not in the current scene.

## What is verified here

Local `official/rsl_rl` is **2.3.1**. `ActorCritic` is an MLP. There is no CNN encoder. The actor reads one flat observation tensor. The critic can read `extras["observations"]["critic"]`. The Isaac task uses that slot for the 2-D material vector. The 4-D Mac-side contract uses the same rule and is not wired into the simulator.

`config/stack.yaml` records this local audit. Isaac Lab, the LocoLab task id, and the rough checkpoint shape are still null because they are not on this Mac.

## Isaac Lab task

The runnable scene is `Isaac-Velocity-Semantic-Unitree-Go2-v0` in `semantic_locomotion/isaaclab_task/`. It inherits the official flat Go2 (`UnitreeGo2FlatEnvCfg`). A front camera on the base (`pos=(0.32, 0.0, 0.06)`) is pitched down so its optical axis intersects the ground 0.45 m ahead of the base. A CNN (`FrontFrictionNet`) maps that RGB frame to `[mu, log10(K)/5]` 0.45 m ahead of the base, and those two numbers are appended to the policy observation. The critic receives the ground-truth `[mu, log10(K)/5]` of that lookahead point from the scene material layout, and `log_foot_material_contacts` logs active foot contacts on the wet slab versus the rock plane into `env.extras["log"]`. `rsl_rl/algorithms/ppo.py` is not modified.

The wet slab is a 10 cm thick kinematic rigid cuboid (`size=(3.2, 4.0, 0.10)`, `pos=(3.0, 0.0, -0.03)`, top surface at `z = +0.02 m`, friction 0.15, compliant stiffness `2e3`). Embedding 8 cm below `z = 0` keeps the top step at 2 cm while providing 10 cm of compliant spring stroke so diagonal-gait foot loads do not bottom out onto the underlying rock plane (`z = 0`, friction 0.80, compliant stiffness `2e5`). This file does not contain a cave mesh.

On the Isaac Lab v2.2.0 machine, with `PYTHONPATH` pointing at this directory:

One locomotion launch trains three policies on the same crossing, then writes `suite_results/summary.json`. `blind` has no pre-contact material, `oracle` gets the true `(mu, log K)` 0.45 m ahead, and `camera` gets the frozen CNN estimate. Spawn is `x` in `[0.95, 1.10]` m, facing the slab, with `v_x` in `[0.4, 0.8]` m/s. Expect about 90 minutes at 512 envs.

```bash
./isaaclab.sh -p /path/to/research/semantic-locomotion/scripts/train_front_friction.py --headless --num_envs 64
./isaaclab.sh -p /path/to/research/semantic-locomotion/scripts/train_semantic_go2.py --headless --num_envs 512
```

`train_front_friction.py` deletes any existing `checkpoints/front_friction.pt` at start, evaluates held-out resets after training, and only writes `checkpoints/front_friction.pt` if `val_mu_mae <= 0.08` (otherwise exiting with code 1). `train_semantic_go2.py` verifies both checkpoint existence and `gate_passed` metadata before launching Isaac Sim. Neither command runs on this Mac.

## Commands

```bash
research/semantic-locomotion/scripts/run_mac_tests.sh
python3 research/semantic-locomotion/scripts/audit_stack.py
```

The slip check in `terrain.slip_proxy` only orders the two materials. It is not a PhysX measurement.
