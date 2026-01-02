from typing import Optional

import numpy as np
from isaacsim.core.prims import SingleArticulation, RigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import get_world_pose_from_relative
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.storage.native import get_assets_root_path

from pxr import Gf

from ARMetaToolPG.assets import ARMT_ASSETS_DATA_DIR, ARMT_ASSETS_DIR

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
        
        policy_path = ARMT_ASSETS_DIR + "/policys/policy_pull_object_rh"
        usd_path = ARMT_ASSETS_DATA_DIR + "/robohabilis/robohabilis_Isaaclab.usd"

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(
            policy_path + "policy.pt",
            policy_path + "env.yaml",
        )

        self._action_scale = 0.5
        self._previous_action = np.zeros(8)
        self._policy_counter = 0

        self.tool = tool
        self.object = object
        self.table = table

    def _compute_observation(self):

        obs = np.zeros(52)

        obs[:16] = self.robot.get_joint_positions() - self.default_pos

        obs[16:32] = self.robot.get_joint_velocities() - self.default_vel

        tool_w_pos, tool_w_quat = self.tool.get_world_poses()
        root_w_pos, root_w_quat = self.robot.get_world_pose()
        tool_b_pos, _ = get_relative_pose(tool_w_pos, tool_w_quat, root_w_pos, root_w_quat)
        obs[32:35] = tool_b_pos

        obj_w_pos, obj_w_quat = self.object.get_world_poses()
        obj_b_pos, _ = get_relative_pose(obj_w_pos, obj_w_quat, root_w_pos, root_w_quat)
        obs[35:38] = obj_b_pos

        obs[38:44] = (0.4, -0.3, 1, 0, 0, 0)

        obs[44:52] = self._previous_action

        return obs

    def forward(self, dt):

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation()
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        # articulation space
        # copy last item for two fingers in order to increase action size from 8 to 9
        # finger positions are absolute positions, not relative to the default position
        self.action[0:8] = self.action[0:8] + self.default_pos[0:8]
        action_input = np.append(self.action, self.action[-1])
        action = ArticulationAction(joint_positions=(action_input * self._action_scale))
        # here action is size 9
        self.robot.apply_action(action)

        self._policy_counter += 1
    
    def initialize(self, physics_sim_view=None) -> None:

        super().initialize(physics_sim_view=physics_sim_view, control_mode="force", set_articulation_props=True)

        self.tool.initialize(physics_sim_view=physics_sim_view)
        self.object.initialize(physics_sim_view=physics_sim_view)
        self.table.initialize(physics_sim_view=physics_sim_view)

        self.robot.set_solver_position_iteration_count(8)
        self.robot.set_solver_velocity_iteration_count(0)
        self.robot.set_stabilization_threshold(0)
        self.robot.set_sleep_threshold(0)

def get_relative_pose(
    pos_a: np.ndarray,
    quat_a: np.ndarray,   # (w, x, y, z)
    pos_b: np.ndarray,
    quat_b: np.ndarray
):
    # Quaternion A
    q_a = Gf.Quatd(quat_a[0], Gf.Vec3d(quat_a[1:]))

    # Quaternion B
    q_b = Gf.Quatd(quat_b[0], Gf.Vec3d(quat_b[1:]))

    # Orientación relativa
    q_rel = q_a.GetInverse() * q_b

    # Posición relativa
    delta_pos = Gf.Vec3d(*(pos_b - pos_a))
    pos_rel = q_a.GetInverse().Transform(delta_pos)

    return (
        np.array([pos_rel[0], pos_rel[1], pos_rel[2]]),
        np.array([q_rel.GetReal(), *q_rel.GetImaginary()])
    )