from typing import Optional

import numpy as np
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_world_pose_from_relative
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.policy.examples.controllers import PolicyController
import isaacsim.core.utils.stage as stage_utils

from ARMetaToolPG.assets import ARMT_ASSETS_DIR, ARMT_ASSETS_DATA_DIR

class UR3eReachPolicy(PolicyController):
    def __init__(
        self,
        prim_path: str,
        table: RigidPrim,
        root_path: Optional[str] = None,
        name: str = "robohabilis",
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        target: np.array = [0.0, 0.30, 0.20, 1, 0, 0, 0]
    ) -> None:
        
        policy_path = ARMT_ASSETS_DIR + "/policys/policy_reach/"
        usd_path = ARMT_ASSETS_DATA_DIR + "/ur3e/ur3e_gripper.usd"
        stage_utils.add_reference_to_stage(usd_path, prim_path)
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(
            policy_path + "policy.pt",
            policy_path + "env.yaml",
        )

        

        self._action_scale = 0.5
        self._previous_action = np.zeros(6)
        self._policy_counter = 0

        self.table = table

    def _compute_observation(self):

        obs = np.zeros(25)

        obs[:6] = self.robot.get_joint_positions() - self.default_pos

        obs[6:12] = self.robot.get_joint_velocities() - self.default_vel

        obs[12:19] = self.target

        obs[19:] = self._previous_action

        return obs

    def forward(self, dt):

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation()
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        # articulation space
        # copy last item for two fingers in order to increase action size from 8 to 9
        # finger positions are absolute positions, not relative to the default position
        self.action = self.action*self._action_scale + self.default_pos
        action = ArticulationAction(joint_positions=(self.action))
        self.robot.apply_action(action)

        self._policy_counter += 1
    
    def initialize(self, physics_sim_view=None) -> None:

        super().initialize(physics_sim_view=physics_sim_view, control_mode="force", set_articulation_props=True)
        
        self.table.initialize(physics_sim_view=physics_sim_view)

        self.robot.set_solver_position_iteration_count(8)
        self.robot.set_solver_velocity_iteration_count(0)
        self.robot.set_stabilization_threshold(0)
        self.robot.set_sleep_threshold(0)