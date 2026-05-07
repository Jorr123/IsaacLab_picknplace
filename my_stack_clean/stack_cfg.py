import math
from dataclasses import MISSING
import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
import isaaclab.utils.math as math_utils

def openvla_proprio_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    hand_pose = asset.data.body_pose_w[:, asset_cfg.body_ids[0]]
    pos = hand_pose[:, :3]
    quat = hand_pose[:, 3:7] 
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    gripper_width = asset.data.joint_pos[:, -2:].sum(dim=1, keepdim=True)
    return torch.cat([pos, roll.unsqueeze(1), pitch.unsqueeze(1), yaw.unsqueeze(1), gripper_width], dim=-1)

def get_relative_transform(env: ManagerBasedRLEnv, handle_a: str, handle_b: str):
    asset_a = env.scene[handle_a]
    asset_b = env.scene[handle_b]
    return asset_a.data.root_pos_w - asset_b.data.root_pos_w

def dist_to_target(env: ManagerBasedRLEnv, asset_cfg_a: SceneEntityCfg, asset_cfg_b: SceneEntityCfg):
    diff = get_relative_transform(env, asset_cfg_a.name, asset_cfg_b.name)
    return torch.norm(diff, dim=-1)

def success_check(env: ManagerBasedRLEnv, threshold: float, 
                  asset_cfg_a: SceneEntityCfg, asset_cfg_b: SceneEntityCfg):
    cube   = env.scene["cube"]
    target = env.scene["target"]

    # Per-env error — shape: [num_envs, 3]
    error  = cube.data.root_pos_w[:, :3] - target.data.root_pos_w[:, :3]

    # Per-env distances — shape: [num_envs]
    xy_err = torch.norm(error[:, :2], dim=-1)
    z_err  = torch.abs(error[:, 2])

    # Per-env boolean — shape: [num_envs]
    is_in_place = (xy_err < threshold) & (z_err < 0.0165)

    # Initialise counter
    if not hasattr(env, "success_counter"):
        env.success_counter = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    # Increment if in place, reset otherwise
    env.success_counter = torch.where(
        is_in_place,
        env.success_counter + 1,
        torch.zeros_like(env.success_counter)
    )

    # Trigger success after 12 consecutive steps (~0.24s at 50Hz)
    return env.success_counter >= 10

@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(1.0, 0.0, 0.0, 0.0), 
            joint_pos={
                "panda_joint1": 0.0143, "panda_joint2": 0.1325, "panda_joint3": -0.0563,
                "panda_joint4": -2.6712, "panda_joint5": 0.0361, "panda_joint6": 2.8951,
                "panda_joint7": 0.7065, "panda_finger_joint.*": 0.04, 
            },
        )
    )
    robot.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
        max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
    )
    
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.045, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0 - 0.05, 0.02)),
    )

    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.15 - 0.05, 0.001)),
    )

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        spawn=sim_utils.PinholeCameraCfg(focal_length=38.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)),
        width=224, height=224, data_types=["rgb"], update_period=0,
        offset=CameraCfg.OffsetCfg(pos=(0.93, 0.00344, 0.445), rot=(0.27131, -0.64716, -0.65462, 0.28115)),
    ) #(x, -w, -z, y)
    
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam", 
        spawn=sim_utils.PinholeCameraCfg(focal_length=23.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)),
        width=224, height=224, data_types=["rgb"], update_period=0,
        offset=CameraCfg.OffsetCfg(pos=(0.03186, 0.00224, 0.05632), rot=(0.71969, -0.07151, -0.05861, 0.68811)),
    )

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

@configclass
class ActionsCfg:
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04}, close_command_expr={"panda_finger_.*": 0.0},
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb", "normalize": False})
        proprio = ObsTerm(func=openvla_proprio_obs, params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")})
        wrist_image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "data_type":"rgb", "normalize": False})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets":True})
    reset_robot_joints_noise = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]), "position_range": (-0.025, 0.025), "velocity_range": (0.0, 0.0)},
    )
    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )
    reset_target_position = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.035, 0.05)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("target"),
        },
    )

@configclass
class RewardsCfg:
    approach_target = RewTerm(
        func=dist_to_target, weight=-2.0,
        params={"asset_cfg_a": SceneEntityCfg("cube"), "asset_cfg_b": SceneEntityCfg("target")}
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=success_check, 
        params={"threshold": 0.055, "asset_cfg_a": SceneEntityCfg("cube"), "asset_cfg_b": SceneEntityCfg("target")},
    )

@configclass
class StackEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 2.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625