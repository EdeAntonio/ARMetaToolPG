# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Add this to the mdp.py file to create a function that ensures the tool is grasped

# def reset_tool_in_grasp(
#         env: ManagerBasedEnv, params=None):
#     """Reset tool to be at the robot's end-effector (grasped position).
    
#     Args:
#         env: The environment instance.
#         params: Additional parameters with keys:
#             - ee_offset: Optional offset from the end-effector position
#             - asset_cfg: Scene entity configuration for the tool
#     """
#     # Default parameters
#     if params is None:
#         params = {}
        
#     # Get the parameters
#     ee_offset = params.get("ee_offset", [0.0, 0.0, 0.0])
#     asset_cfg = params.get("asset_cfg", SceneEntityCfg("tool"))
    
#     # Get the tool asset
#     tool_asset = env.scene.get_asset(asset_cfg.name)
#     tool_bodies = asset_cfg.get_bodies_in_asset(tool_asset)
    
#     # Get the end-effector position and orientation
#     ee_frame = env.scene.ee_frame
#     ee_pos, ee_rot = ee_frame.get_world_pose()
    
#     # Apply offset to the position (in ee frame)
#     offset_pos = sim_utils.transform_points(ee_offset, ee_pos, ee_rot)
    
#     # Set the tool position and orientation to match the end-effector
#     for env_idx in range(env.num_envs):
#         for body in tool_bodies:
#             # Set the position and orientation
#             tool_asset.set_rigid_body_pose(body, offset_pos[env_idx], ee_rot[env_idx], env_indices=env_idx)
            
#             # Zero out velocities
#             tool_asset.set_rigid_body_linear_velocity(body, [0.0, 0.0, 0.0], env_indices=env_idx)
#             tool_asset.set_rigid_body_angular_velocity(body, [0.0, 0.0, 0.0], env_indices=env_idx)
    
#     return


# def reset_tool_in_grasp(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     ee_offset: list[float] = [0.0, 0.0, 0.0],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
# ):
#     """Reset tool to be at the robot's end-effector (grasped position).
    
#     This function positions the tool at the robot's end-effector with an optional offset.
#     The tool's position, orientation, and velocities are updated to match the end-effector,
#     effectively making it appear grasped by the robot.
    
#     Args:
#         env: The environment instance.
#         env_ids: Tensor containing environment indices to apply the reset to.
#         ee_offset: Optional offset from the end-effector position in the end-effector's local frame.
#         asset_cfg: Scene entity configuration for the tool.
#     """
#     # Extract the used quantities (to enable type-hinting)
#     tool_asset: RigidObject = env.scene[asset_cfg.name]
#     tool_bodies = asset_cfg.get_bodies(tool_asset)
    
#     # Get the end-effector position and orientation
#     ee_frame: FrameTransformer = env.scene.ee_frame
#     ee_pos, ee_rot = ee_frame.get_world_pose(env_ids)
    
#     # Apply offset to the position (in ee frame)
#     offset_tensor = torch.tensor(ee_offset, device=env_ids.device).expand(len(env_ids), 3)
#     offset_pos = transform_utils.transform_points(offset_tensor, ee_pos, ee_rot)
    
#     # Set zero velocities
#     zero_lin_vel = torch.zeros_like(ee_pos)
#     zero_ang_vel = torch.zeros_like(ee_pos)
    
#     # Set the tool position and orientation to match the end-effector for each body
#     for body in tool_bodies:
#         # Set the position and orientation
#         tool_asset.set_rigid_body_pose(body, offset_pos, ee_rot, env_indices=env_ids)
        
#         # Zero out velocities
#         tool_asset.set_rigid_body_linear_velocity(body, zero_lin_vel, env_indices=env_ids)
#         tool_asset.set_rigid_body_angular_velocity(body, zero_ang_vel, env_indices=env_ids)