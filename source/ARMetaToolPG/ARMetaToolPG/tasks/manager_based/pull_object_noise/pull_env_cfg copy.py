# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from MT_ext.assets import MT_ASSETS_DATA_DIR

from . import mdp

##
# Scene definition
##


@configclass
class PullObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the pull object with tool scene with a robot, a pulling tool, and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target Tool: will be populated by agent env cfg
    tool: RigidObjectCfg | DeformableObjectCfg = MISSING

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0, 0.055], rot=[0, 0, 0, 1]), #pos and rot needs to be accesible for tool generation process
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale = (1.0,1.0,1.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.09, 0.56, -0.825), rot=(0.70711, 0.0, 0.0, -0.70711)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{MT_ASSETS_DATA_DIR}/Table/MT_table.usd",
            rigid_props = RigidBodyPropertiesCfg(
                kinematic_enabled = True,
            )
            
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.825]),
        # init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    #Adding Contact sensors for the object and tool 
    object_contact_sensor: ContactSensorCfg = MISSING
    tool_contact_sensor: ContactSensorCfg = MISSING

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    # tool_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name= MISSING, # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(-0.3, -0.2), pos_y=(0.0, 0.0), pos_z=(0.0, 0.0), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )
    
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.3), pos_y=(-0.2, -0.2), pos_z=(0.0, 0.0), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        tool_position = ObsTerm(func=mdp.tool_position_in_robot_root_frame)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    # Add this to ensure the tool is grasped after the general reset
    # reset_tool_to_grasp = EventTerm(
    #     func=mdp.reset_tool_in_grasp,
    #     mode="reset",
    #     params={
    #         "ee_offset": [0.0, 0.0, 0.05],  # Offset from the end-effector (may need adjustment based on tool)
    #         "asset_cfg": SceneEntityCfg("tool"),
    #     },
    # )

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.05, 0.05), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reaching_tool = RewTerm(func=mdp.tool_ee_distance, params={"std": 0.1}, weight=1.0)
    
    # reaching_object = RewTerm(func=mdp.object_tool_distance, params={"std": 0.3}, weight=20.0)

    # lifting_tool = RewTerm(func=mdp.tool_is_lifted, params={"minimal_height": 0.05}, weight=5.0)

    # grasping_tool = RewTerm(func=mdp.tool_is_grasped, params={"std": 0.1}, weight=8.0)
    # # grasping_tool = RewTerm(func=mdp.tool_is_grasped, weight=5.0)

    # pulling_object = RewTerm(func=mdp.object_is_pulled, params={"std": 0.2}, weight=10.0)

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "command_name": "object_pose"},
    #     weight=30.0,
    # )

    reaching_tool = RewTerm(func=mdp.tool_ee_distance, params={"std": 0.1}, weight=1.0)
    
    reaching_object = RewTerm(func=mdp.object_tool_distance, params={"std": 0.3}, weight=20.0)

    lifting_tool = RewTerm(func=mdp.tool_is_lifted, params={"minimal_height": 0.05}, weight=5.0)

    grasping_tool = RewTerm(func=mdp.tool_is_grasped, params={"std": 0.1}, weight=8.0)
    # grasping_tool = RewTerm(func=mdp.tool_is_grasped, weight=5.0)

    pulling_object = RewTerm(func=mdp.object_is_pulled, params={"std": 0.2}, weight=10.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "command_name": "object_pose"},
        weight=40.0,
    )
    # TODO: Check if this implementation is working correctly
    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.1, "des_pos": [0.3, -0.2, 0.0]}, #"des_ori": quaternion
    #     weight=30.0,
    # )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "command_name": "object_pose"},
        weight=5.0,
    )
    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "goal_pos": [0.3, -0.2, 0.0], "goal_rot": [0.0, 0.0, 0.0]},
    #     weight=5.0,
    # )
    # action penalty

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    tool_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("tool")}
    )

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class PullEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: PullObjectTableSceneCfg = PullObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
