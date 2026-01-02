import isaacsim.core.utils.stage as stage_utils
import numpy as np
import omni
from isaacsim.core.prims import SingleArticulation, RigidPrim
from ARMetaToolPG.assets.base_sample import BaseSample

from ARMetaToolPG.assets import ARMT_ASSETS_DATA_DIR
from ARMetaToolPG.assets.policys.policy_controllers.robohabilis import RobohabilisPullObjectPolicy

import carb

NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"

class RoboHabilis(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 400.0
        self._world_settings["rendering_dt"] = 1.0 / 60.0
    
    def setup_scene(self) -> None:

        self.get_world().scene.add_default_ground_plane(
            z_position= -0.825,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )

        object_prim_path = "/World/object"
        object_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        object_name = "object"
        object_position = np.array([0.6, -0.4, 0.055])
        object_rotation = np.array([0.0, 0.0, 0.0, 1.0])

        stage_utils.add_reference_to_stage(object_usd_path, object_prim_path)
        self.object = RigidPrim(
            prim_paths_expr=object_prim_path, name=object_name, positions=object_position, orientations=object_rotation,
            scales=(1.0, 1.0, 1.0)
        )

        tool_prim_path = "/World/tool"
        tool_usd_path = f"{ARMT_ASSETS_DATA_DIR}/Tool/big_hook.usd"
        tool_name = "tool"
        tool_position = np.array([0.143, -0.3161, 0.008])
        tool_rotation = np.array([1.0, 0.0, 0.0, 0.0])

        stage_utils.add_reference_to_stage(tool_usd_path, tool_prim_path)
        self.tool = RigidPrim(
            prim_paths_expr=tool_prim_path, name=tool_name, positions=tool_position, orientations=tool_rotation,
            scales=(0.001, 0.001, 0.001)
        )

        table_prim_path = "/World/table"
        table_usd_path = f"{ARMT_ASSETS_DATA_DIR}/Table/MT_table.usd"
        table_name = "table"
        table_position = np.array([-0.09, 0.56, -0.825])
        table_rotation = np.array([0.70711, 0.0, 0.0, -0.70711])

        stage_utils.add_reference_to_stage(table_usd_path, table_prim_path)
        self.table = RigidPrim(
            prim_paths_expr=table_prim_path, name=table_name, positions=table_position, orientations=table_rotation
        )

        self.robohabilis = RobohabilisPullObjectPolicy(
            prim_path="/World/robohabilis", name="robohabilis", position=np.array([0, 0, 0]), tool=self.tool, 
            object=self.object, table=self.table
        )

        timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._timeline_timer_callback_fn
        )
    
    async def setup_post_load(self) -> None:
        self._physics_ready = False
        self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

        await self.get_world().play_async()
    

    async def setup_post_reset(self) -> None:
        self._physics_ready = False
        self.robohabilis.previous_action = np.zeros(9)
        await self.get_world().play_async()

    def on_physics_step(self, step_size) -> None:
        if self._physics_ready:
            self.robohabilis.forward(step_size)
        else:
            self._physics_ready = True
            self.robohabilis.initialize()
            self.robohabilis.post_reset()
            self.robohabilis.robot.set_joints_default_state(self.franka.default_pos)

    def _timeline_timer_callback_fn(self, event) -> None:
        if self.robohabilis:
            self._physics_ready = False

    def world_cleanup(self):
        world = self.get_world()
        self._event_timer_callback = None
        if world.physics_callback_exists("physics_step"):
            world.remove_physics_callback("physics_step")



