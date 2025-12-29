# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
# from isaaclab.sim.spawners.wrappers.wrappers_cfg import MultiUsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ARMetaToolPG.tasks.manager_based.pull_object import mdp
from ARMetaToolPG.tasks.manager_based.pull_object.pull_env_cfg import PullEnvCfg 
from ARMetaToolPG.utils.convert_mesh import *
from ARMetaToolPG.assets import ARMT_ASSETS_DATA_DIR

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from ARMetaToolPG.assets.robots.robohabilis import L_ROBOHABILIS_CFG, ROBOHABILIS_CFG


@configclass
class RobohabilisCubePullEnvCfg(PullEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Allow for multiobject simulation
        # self.scene.replicate_physics= False
        # Set Robohabilis as robot
        #self.scene.robot = L_ROBOHABILIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = ROBOHABILIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Set actions for the specific robot type (Robohabilis right arm)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["l_shoulder_pan_joint","l_shoulder_lift_joint","l_elbow_joint","l_wrist_1_joint","l_wrist_2_joint","l_wrist_3_joint"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["l_.*_finger_joint"],
            open_command_expr={"l_.*_finger_joint": 0.00},
            close_command_expr={"l_.*_finger_joint": -0.013},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "l_gripper_body"

        # Define tool as rigid object
        self.scene.tool = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tool",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.143, -0.3161, 0.008], rot=[1, 0, 0, 0]), # pos=[0.273, -0.3181, 0.005],rot=[0.08716, 0, 0, -0.99619]
            spawn=UsdFileCfg(
                usd_path=f"{ARMT_ASSETS_DATA_DIR}/Toolpuedo/big_hook.usd",
                scale=(0.001, 0.001, 0.001),
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
        
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/l_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/l_gripper_body",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.object.spawn.activate_contact_sensors = True
        self.scene.tool.spawn.activate_contact_sensors = True

        self.scene.object_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Tool",
            ],
            track_air_time=True,
            track_pose=True,
            update_period=0.0,
            debug_vis=True,
        )
        self.scene.tool_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Tool",
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Robot/l_left_finger",
                "{ENV_REGEX_NS}/Robot/l_right_finger",
            ],
            track_air_time=True,
            track_pose=True,
            update_period=0.0,
            debug_vis=True,
        )


@configclass
class RobohabilisCubePullEnvCfg_PLAY(RobohabilisCubePullEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
