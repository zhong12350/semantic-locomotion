"""Go2 velocity task: front camera estimates friction and stiffness before contact.

Parent is the official flat Go2 task in Isaac Lab v2.2.0. The plane is stiff
rock. A darker slab ahead of the robot is the wet, compliant patch. The image
never enters ``rsl_rl/algorithms/ppo.py``. A separate CNN writes two numbers
into the policy observation.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)

from semantic_locomotion.isaaclab_task.observations import (
    camera_mu_k,
    ground_truth_mu_k,
)
from semantic_locomotion.physics_patches import (
    CAMERA_FOCAL_LENGTH_MM,
    CAMERA_HORIZONTAL_APERTURE_MM,
    CAMERA_OFFSET_POS,
    CROSSING_HEADING,
    CROSSING_LIN_VEL_X,
    CROSSING_LIN_VEL_Y,
    CROSSING_SPAWN_X,
    CROSSING_SPAWN_Y,
    CROSSING_SPAWN_YAW,
    K_ROCK,
    K_WET,
    MU_ROCK,
    MU_WET,
    WET_SLAB_POS,
    WET_SLAB_SIZE,
    camera_ros_quat_wxyz,
)

TASK_ID = "Isaac-Velocity-Semantic-Unitree-Go2-v0"


@configclass
class CriticCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)
    ground_truth_mu_k = ObsTerm(func=ground_truth_mu_k)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UnitreeGo2SemanticEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 512
        self.scene.env_spacing = 8.0

        rock = self.scene.terrain.physics_material
        rock.static_friction = MU_ROCK
        rock.dynamic_friction = 0.70
        rock.restitution = 0.0
        rock.friction_combine_mode = "min"
        rock.compliant_contact_stiffness = K_ROCK
        rock.compliant_contact_damping = 2.0e3
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.42, 0.28),
            roughness=0.55,
        )

        # 10 cm thick compliant slab embedded 8 cm into the ground so its top is
        # 2 cm above the rock plane (top z = 0.02 m, bottom z = -0.08 m).
        # With compliant_contact_stiffness = 2e3 N/m, a diagonal-gait foot load
        # (~74 N) compresses the surface by ~3.7 cm, staying well above z = 0 so
        # the foot never bottoms out onto the stiff rock plane underneath.
        self.scene.wet_patch = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/WetPatch",
            spawn=sim_utils.CuboidCfg(
                size=WET_SLAB_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=MU_WET,
                    dynamic_friction=0.12,
                    restitution=0.0,
                    friction_combine_mode="min",
                    compliant_contact_stiffness=K_WET,
                    compliant_contact_damping=2.0e2,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 0.22, 0.18),
                    roughness=0.95,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=WET_SLAB_POS),
        )

        # Pitched down so the optical axis intersects the ground 0.45 m ahead of the base.
        self.scene.front_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=CAMERA_OFFSET_POS,
                rot=camera_ros_quat_wxyz(),
                convention="ros",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=CAMERA_FOCAL_LENGTH_MM,
                focus_distance=400.0,
                horizontal_aperture=CAMERA_HORIZONTAL_APERTURE_MM,
                clipping_range=(0.05, 8.0),
            ),
            width=64,
            height=64,
            data_types=["rgb"],
            update_period=self.decimation * self.sim.dt,
        )

        self.observations.policy.precontact_mu_k = ObsTerm(func=camera_mu_k)
        self.observations.critic = CriticCfg()


def configure_wet_crossing(env_cfg, *, condition: str) -> None:
    """Same wet-slab crossing for three actor inputs.

    ``blind`` sees no pre-contact material. ``oracle`` sees the true (mu, log K)
    of the point 0.45 m ahead. ``camera`` sees the frozen CNN estimate of that
    same vector. The critic always sees the ground truth. Spawn and commands match.
    """
    if condition not in ("blind", "oracle", "camera"):
        raise ValueError(f"condition must be blind, oracle, or camera, got {condition}")

    pose = env_cfg.events.reset_base.params["pose_range"]
    pose["x"] = CROSSING_SPAWN_X
    pose["y"] = CROSSING_SPAWN_Y
    pose["yaw"] = CROSSING_SPAWN_YAW

    command = env_cfg.commands.base_velocity
    command.rel_standing_envs = 0.0
    command.heading_command = True
    command.ranges.lin_vel_x = CROSSING_LIN_VEL_X
    command.ranges.lin_vel_y = CROSSING_LIN_VEL_Y
    command.ranges.heading = CROSSING_HEADING

    if condition == "oracle":
        env_cfg.observations.policy.precontact_mu_k = ObsTerm(func=ground_truth_mu_k)
        env_cfg.scene.front_camera = None
    elif condition == "camera":
        env_cfg.observations.policy.precontact_mu_k = ObsTerm(func=camera_mu_k)
    else:
        env_cfg.observations.policy.precontact_mu_k = None
        env_cfg.scene.front_camera = None


@configclass
class UnitreeGo2SemanticPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.max_iterations = 1500
        self.experiment_name = "unitree_go2_semantic"
        self.empirical_normalization = False
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


def register() -> str:
    import gymnasium as gym

    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "semantic_locomotion.isaaclab_task.go2_semantic_cfg:UnitreeGo2SemanticEnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "semantic_locomotion.isaaclab_task.go2_semantic_cfg:UnitreeGo2SemanticPPORunnerCfg"
                ),
            },
        )
    return TASK_ID
