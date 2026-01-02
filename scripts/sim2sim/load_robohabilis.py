import asyncio

import numpy as np

import omni

from isaacsim.core.api.world import World
from isaacsim.core.prims import RigidPrim
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api import SimulationContext


from ARMetaToolPG.assets import ARMT_ASSETS_DATA_DIR
from ARMetaToolPG.assets.policys.policy_controllers.robohabilis import RobohabilisPullObjectPolicy


import carb

NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"

settings=carb.settings.get_settings()
settings.set("/log/level", "error")

async def load_robohabilis():
    try:
        print("Funcion llamada\n")
        robohabilis_task = RoboHabilisTask()
        robohabilis_task.set_up_scene()
        await robohabilis_task.load_world_async()
        print("Carga completada\n")
    except Exception as e:
        import traceback
        print("EXCEPCIÓN EN load_robohabilis")
        traceback.print_exc()



class RoboHabilisTask(object):
    def __init__(self):
        if World.instance():
            World.instance().clear_instance()

        self.world=World(
            physics_dt=1.0/60.0,
            rendering_dt=1.0/60.0,
            stage_units_in_meters=1
        )

    def set_up_scene(self):
        self.world.scene.add_default_ground_plane(
            z_position= -0.825,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )

        table_prim_path = "/World/table"
        table_usd_path = f"{ARMT_ASSETS_DATA_DIR}/Table/MT_table.usd"
        table_name = "table"
        table_position = np.array([[-0.09, 0.56, -0.825]], dtype=float)
        table_rotation = np.array([[0.70711, 0.0, 0.0, -0.70711]], dtype=float)

        stage_utils.add_reference_to_stage(table_usd_path, table_prim_path)
        table = RigidPrim(
            prim_paths_expr=table_prim_path, name=table_name, positions=table_position, orientations=table_rotation
        )
        self.world.scene.add(table)

        tool_prim_path = "/World/tool"
        tool_usd_path = f"{ARMT_ASSETS_DATA_DIR}/Tool/big_hook.usd"
        tool_name = "tool"
        tool_position = np.array([[0.143, -0.3161, 0.008]], dtype=float)
        tool_rotation = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
        tool_scale = np.array([[0.001, 0.001, 0.001]], dtype=float)

        stage_utils.add_reference_to_stage(tool_usd_path, tool_prim_path)
        tool = RigidPrim(
            prim_paths_expr=tool_prim_path, name=tool_name, positions=tool_position, orientations=tool_rotation,
            scales=tool_scale
        )
        self.world.scene.add(tool)

        object_prim_path = "/World/object"
        object_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        object_name = "object"
        object_position = np.array([[0.6, -0.4, 0.055]], dtype=float)
        object_rotation = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
        object_scales=np.array([[0.0005, 0.0005, 0.0005]], dtype=float)

        stage_utils.add_reference_to_stage(object_usd_path, object_prim_path)
        object = RigidPrim(
            prim_paths_expr=object_prim_path, name=object_name, positions=object_position, orientations=object_rotation,
            scales=object_scales
        )
        self.world.scene.add(object)

        self.robohabilis = RobohabilisPullObjectPolicy(
            prim_path="/World/robohabilis", name="robohabilis", position=np.array([[0, 0, 0]]), tool=tool, 
            object=object, table=table
        )

        print("Escena creada.\n")

    async def load_world_async(self):
        await self.world.initialize_simulation_context_async()
        print("Se trata de cargar el mundo.\n")
        await self.world.reset_async()
        print("Mundo reseteado.\n")
        await self.world.pause_async()
        print("Mundo parado.\n")
        await self.setup_post_load()
    
    async def setup_post_load(self) -> None:
        print("Se trata de grabar la función recurrente")
        self._physics_ready = False

        sim_ctx = SimulationContext.instance()
        sim_ctx.add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        print("Función recurrente grabada.")

        # Esperar un frame para que PhysX registre el robot
        await asyncio.sleep(0.1)

        # Ahora inicializar el robot
        self.robohabilis.initialize()
        self.robohabilis.post_reset()
        self.robohabilis.robot.set_joints_default_state(self.robohabilis.default_pos)

        await self.world.play_async()

    def on_physics_step(self, step_size) -> None:
        self.robohabilis.forward(step_size)


asyncio.ensure_future(load_robohabilis())