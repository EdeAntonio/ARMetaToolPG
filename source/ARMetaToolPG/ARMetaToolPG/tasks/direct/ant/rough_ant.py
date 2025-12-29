# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_assets.robots.ant import ANT_CFG


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
import torch

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

RANDOM_ROUGH_CFG = terrain_gen.TerrainGeneratorCfg(
    size = (100.0, 100.0),
    num_rows = 1,
    num_cols = 1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains = {
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion= 1.0, noise_range=(0.02, 0.10), noise_step=0.02
        )
    }
)

@configclass
class RoughAntEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 0.5
    action_space = 8
    observation_space = 36
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    
    # terreno
    terrain = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group= -1,
        terrain_generator = RANDOM_ROUGH_CFG,
        max_init_terrain_level=5,
        physics_material= sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

        # Robot configuration
    robot: ArticulationCfg = ANT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [15, 15, 15, 15, 15, 15, 15, 15]

        # Camara
    camera: TiledCameraCfg = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Robot/torso/FrontCamera",
    update_period= 0.1,
    height = 64,
    width= 64,
    data_types= ["rgb", "depth"],
    spawn= PinholeCameraCfg(
        focal_length = 24.0, focus_distance = 400, clipping_range=(0.1, 20)
    ),
    offset= TiledCameraCfg.OffsetCfg(pos = (0.3, 0, 0), rot = (0.9239,0,-0.3827,0), convention = "world")
)
    

    # scene
    scene = InteractiveSceneCfg(
        num_envs=4096, 
        env_spacing=4.0, 
        replicate_physics=True, 
        clone_in_fabric=True,
    )
    
    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.2

    death_cost: float = -2.0
    termination_height: float = 0.41

    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1


class RoughAntEnv(LocomotionEnv):
    cfg: RoughAntEnvCfg

    def __init__(self, cfg: RoughAntEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.cont = 0

    def _setup_scene(self):
        super()._setup_scene()
        self.camera = TiledCamera(self.cfg.camera)

    def _get_observations(self):
        obs = torch.cat(
            (
                self.torso_position[:, 2].view(-1, 1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions
            ),
            dim=-1,
        )
        self.cont += 1
        if self.cont == 25:
            self.cont= 0
            matrix = self.camera.data.output["depth"].shape
            print(matrix)
        observations = {"policy": obs}
        return observations

        