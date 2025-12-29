# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, ContactSensor, ContactSensorCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def tool_is_grasped(
#     env: ManagerBasedRLEnv,
#     std: float, 
#     tool_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
#     tool_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tool_contact_sensor"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Reward the agent for grasping the object."""
#     tool: RigidObject = env.scene[tool_cfg.name]
#     robot: Articulation = env.scene[robot_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     tool_contact_sensor: ContactSensor = env.scene[tool_contact_sensor_cfg.name]
#     # Check active contact (current_contact_time > 0)
#     contact_active = (tool_contact_sensor.data.current_contact_time.squeeze(-1) > 0).float()
#     # Get positions of tool and end effector
#     tool_pos_w = tool.data.root_pos_w # Target object position: (num_envs, 3)
#     ee_w = ee_frame.data.target_pos_w[..., 0, :]     # End-effector position: (num_envs, 3)
#     tool_ee_distance = torch.norm(tool_pos_w - ee_w, dim=1) # Distance of the end-effector to the object: (num_envs,)
#     # Compute distance-based reward component
#     distance_reward = 1 - torch.tanh(tool_ee_distance / std)
#     # Combine contact presence and proximity reward
#     total_reward = contact_active * distance_reward
    
#     return total_reward

def tool_is_grasped(
    env: ManagerBasedRLEnv,
    std: float, 
    tool_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
    tool_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tool_contact_sensor"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    diff_threshold: float = 0.06,
    # gripper_open_val: torch.tensor = torch.tensor([0.04]), # FRANKA EMIKA
    gripper_open_val: torch.tensor = torch.tensor([0.013]), # ROBOHABILIS
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Reward the agent for grasping the object."""
    tool: RigidObject = env.scene[tool_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    tool_pos = tool.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(tool_pos - ee_pos, dim=1) 

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
    )  
    
#    grasped = torch.logical_and(
#        pose_diff < diff_threshold,
#        torch.sub(torch.abs(robot.data.joint_pos[:, -1]), gripper_open_val.to(env.device)) < gripper_threshold,
#    )
#    grasped = torch.logical_and(
#        grasped, torch.sub(torch.abs(robot.data.joint_pos[:, -1]), gripper_open_val.to(env.device)) < gripper_threshold
#    )  

    # tool_contact_sensor: ContactSensor = env.scene[tool_contact_sensor_cfg.name]
    # # Check active contact (current_contact_time > 0)
    # contact_active = (tool_contact_sensor.data.current_contact_time.squeeze(-1) > 0).float()
    # # Get positions of tool and end effector
 
    # # Compute distance-based reward component
    # distance_reward = 1 - torch.tanh(tool_ee_distance / std)
    # # Combine contact presence and proximity reward
    # total_reward = contact_active * distance_reward
    
    return grasped

def object_is_pulled(
    env: ManagerBasedRLEnv,
    std: float, 
    tool_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_contact_sensor"),
    tool_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tool_contact_sensor"),
) -> torch.Tensor:
    """Reward the agent for grasping the object."""
    tool: RigidObject = env.scene[tool_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_contact_sensor: ContactSensor = env.scene[object_contact_sensor_cfg.name]
    tool_contact_sensor: ContactSensor = env.scene[tool_contact_sensor_cfg.name]
    # minimal_distance = 0.1
    # Check robot tool active contact (current_contact_time > 0)
    tool_contact_active = (torch.norm(tool_contact_sensor.data.net_forces_w, -1) > 0.01).float()
    # Check tool object active contact (current_contact_time > 0)
    object_contact_active = (torch.norm(object_contact_sensor.data.net_forces_w, -1) > 0.01).float()
    # Get positions of tool and object
    tool_pos_w = tool.data.root_pos_w # Target object position: (num_envs, 3)
    object_pos_w = object.data.root_pos_w # End-effector position: (num_envs, 3)
    object_tool_distance = torch.norm(tool_pos_w - object_pos_w, dim=1) # Distance of the end-effector to the object: (num_envs,)
    # Determine if the object is near the tool
    # object_tool_distance = torch.where(object_tool_distance > minimal_distance, object_tool_distance, 0.0)
    valid_contact = (tool_contact_active * object_contact_active).bool()
    object_tool_distance = torch.where(valid_contact, 0.0, object_tool_distance)
    # Compute distance-based reward component
    distance_reward = 1 - torch.tanh(object_tool_distance / std)
    #Combine contact presence and proximity reward
    total_reward = object_contact_active * distance_reward * tool_contact_active
    # total_reward = object_contact_active * tool_contact_active
    return total_reward

def tool_is_lifted(
    env: ManagerBasedRLEnv, 
    minimal_height: float, 
    tool_cfg: SceneEntityCfg = SceneEntityCfg("tool")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[tool_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def tool_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    tool_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    # tool_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("tool_contact_sensor"),
) -> torch.Tensor:
    """Reward the agent for reaching the tool using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    tool: RigidObject = env.scene[tool_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    tool_pos_w = tool.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    test= tool_pos_w - ee_w
    # Distance of the end-effector to the object: (num_envs,)
    tool_ee_distance = torch.norm(test, dim=1)

    return 1 - torch.tanh(tool_ee_distance / std)

def object_tool_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    tool_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
) -> torch.Tensor:
    """Reward the agent for reaching the object with the tool using tanh-kernel."""
    # minimal_distance = 0.15
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    tool: RigidObject = env.scene[tool_cfg.name]
    # Target object position: (num_envs, 3)
    object_pos_w = object.data.root_pos_w#[...,:2]
    # End-effector position: (num_envs, 3)
    tool_pos_w = tool.data.root_pos_w#[...,:2]
    # Distance of the end-effector to the object: (num_envs,)
    object_tool_distance = torch.norm(tool_pos_w - object_pos_w, dim=1)

    # object_tool_distance = torch.where(object_tool_distance > minimal_distance, object_tool_distance, 0.0)

    return  1.0 - torch.tanh(object_tool_distance / std)


# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     # command_name: str,
#     tool_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     des_pos: tuple[float, float, float] = (0.4, 0.1, 0.0),
#     des_ori: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
# ) -> torch.Tensor:
#     """Reward the agent for pushing the object to the goal position and orientation within the table plane using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     tool: RigidObject = env.scene[tool_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     # command = env.command_manager.get_command(command_name)
#     # compute the desired position and orientation in the world frame
#     des_pos_b = torch.zeros((env.num_envs, 3), device=env.device)
#     des_pos_b [:,:3]= torch.tensor(des_pos, device=env.device)

#     des_ori_b = torch.zeros((env.num_envs, 4), device=env.device)
#     des_ori_b[:, :4] = torch.tensor(des_ori, device=env.device)
#     # des_pos_b = command[:, :3]
#     # des_ori_b = command[:, 3:7]
#     des_pos_w, _ = combine_frame_transforms(tool.data.root_state_w[:, :3], tool.data.root_state_w[:, 3:7], des_pos_b, des_ori_b)
#     # distance of the object to the desired position: (num_envs,)
#     # pos_distance = torch.norm(des_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
#     # orientation difference between the object and the desired orientation: (num_envs,)
#     # ori_distance = torch.norm(des_ori_w - object.data.root_state_w[:, 3:7], dim=1)
#     pos_distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)

#     # reward based on position and orientation distance
#     return (1 - torch.tanh(pos_distance / std))

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (1 - torch.tanh(distance / std))


# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     command_name: str,
#     tool_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for pushing the object to the goal position and orientation within the table plane using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     tool: RigidObject = env.scene[tool_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position and orientation in the world frame
#     des_pos_b = command[:, :3]
#     des_ori_b = command[:, 3:7]
#     des_pos_w, des_ori_w = combine_frame_transforms(tool.data.root_state_w[:, :3], tool.data.root_state_w[:, 3:7], des_pos_b, des_ori_b)
#     # distance of the object to the desired position: (num_envs,)
#     pos_distance = torch.norm(des_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
#     # orientation difference between the object and the desired orientation: (num_envs,)
#     ori_distance = torch.norm(des_ori_w - object.data.root_state_w[:, 3:7], dim=1)
#     # reward based on position and orientation distance
#     return (1 - torch.tanh(pos_distance / std)) * (1 - torch.tanh(ori_distance / std))

def touch_desk(env: ManagerBasedRLEnv,  
    robot_desk_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot_desk_sensor")):
    robot_desk_contact_sensor: ContactSensor = env.scene[robot_desk_contact_sensor_cfg.name]
    #   Check active contact (current_contact_time > 0)
    contact_active = (torch.norm(robot_desk_contact_sensor.data.net_forces_w, -1) > 0.01).bool()
    return torch.where(contact_active, 1, 0)