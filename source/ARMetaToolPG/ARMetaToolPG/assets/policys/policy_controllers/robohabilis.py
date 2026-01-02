from typing import Optional

import numpy as np
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_world_pose_from_relative
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.policy.examples.controllers import PolicyController
import isaacsim.core.utils.stage as stage_utils

from pxr import Gf

from ARMetaToolPG.assets import ARMT_ASSETS_DIR, ARMT_ASSETS_DATA_DIR

class RobohabilisPullObjectPolicy(PolicyController):
    def __init__(
        self,
        prim_path: str,
        tool: RigidPrim,
        table: RigidPrim,
        object: RigidPrim,
        root_path: Optional[str] = None,
        name: str = "robohabilis",
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        
        policy_path = "/home/eantonio/ARMetaToolPG/source/ARMetaToolPG/ARMetaToolPG/assets/policys/policy_pull_object_rh/"
        usd_path = ARMT_ASSETS_DATA_DIR + "/robohabilis/robohabilis_Isaaclab.usd"
        print(usd_path)
        stage_utils.add_reference_to_stage(usd_path, prim_path)
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(
            policy_path + "policy.pt",
            policy_path + "env.yaml",
        )

        

        self._action_scale = 0.5
        self._previous_action = np.zeros(7)
        self._policy_counter = 0

        self.tool = tool
        self.object = object
        self.table = table

    def _compute_observation(self):

        obs = np.zeros(52)

        obs[:16] = self.robot.get_joint_positions() - self.default_pos

        obs[16:32] = self.robot.get_joint_velocities() - self.default_vel

        tool_w_pos, _ = self.tool.get_world_poses()
        obs[32:35] = tool_w_pos

        obj_w_pos, _ = self.object.get_world_poses()
        obs[35:38] = obj_w_pos

        obs[38:45] = (0.4, -0.3, 0, 1, 0, 0, 0)

        obs[45:52] = self._previous_action

        return obs

    def forward(self, dt):

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation()
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        # articulation space
        # copy last item for two fingers in order to increase action size from 8 to 9
        # finger positions are absolute positions, not relative to the default position
        self.action[0:6] = self.action[0:6]*self._action_scale + self.default_pos[0:6]
        self.joint_pos_tg = np.zeros(16)
        if(self.action[6]<0):
            self.joint_pos_tg[6:8] = np.array([-0.013, -0.013])
        else:
            self.joint_pos_tg[6:8] = np.zeros(2)
        
        self.joint_pos_tg[0:6] = self.action[0:6]
        self.joint_pos_tg[8:16] = self.default_pos[8:16]
        action = ArticulationAction(joint_positions=(self.joint_pos_tg))
        self.robot.apply_action(action)

        self._policy_counter += 1
    
    def initialize(self, physics_sim_view=None) -> None:

        super().initialize(physics_sim_view=physics_sim_view, control_mode="force", set_articulation_props=True)
        self.robot.set_joint_positions(self.default_pos)

        self.tool.initialize(physics_sim_view=physics_sim_view)
        self.object.initialize(physics_sim_view=physics_sim_view)
        self.table.initialize(physics_sim_view=physics_sim_view)

        self.robot.set_solver_position_iteration_count(8)
        self.robot.set_solver_velocity_iteration_count(0)
        self.robot.set_stabilization_threshold(0)
        self.robot.set_sleep_threshold(0)
