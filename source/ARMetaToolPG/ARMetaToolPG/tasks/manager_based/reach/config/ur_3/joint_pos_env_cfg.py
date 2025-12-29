# Script para la creación del entorno para la herramienta IsaacLab
# Objetivo: Ejercicio de reach para robot UR3
# Autor: Enrique de Antonio
# Basado en ejercicio de IsaacLab Reach UR10. Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).

import math
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

import ARMetaToolPG.tasks.manager_based.reach.mdp as mdp
from ARMetaToolPG.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg

# Configuración del robot
from ARMetaToolPG.assets.robots.ur3_configuration import UR3_CFG

# Configuración del entorno
@configclass
class UR3ReachEnvCfg(ReachEnvCfg):

    # Inicialización del entorno
    def __post_init__(self):
        # Ejecutar inicialización por defecto.
        super().__post_init__()

        # Incluir nuestro correspondiente robot, UR3
        self.scene.robot = UR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Cambiar el rango de la posición inicial
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)

        # Ajustar el nombre para la referencia de las recompensas. 
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_center"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_center"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_center"]
        
        # Ajustar el nombre para la referencia de las acciones
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )

        # Ajustar el nombre para la referencia de las comandas
        self.commands.ee_pose.body_name = "gripper_center"
        # self.commands.ee_pose.ranges.pitch = (math.pi/2, math.pi/2)
        self.commands.ee_pose.ranges.pos_x = (0.15, 0.40)
        self.commands.ee_pose.ranges.pos_z = (0.15, 0.30)
        self.commands.ee_pose.ranges.pos_y = (-0.30, 0.20)

        # Modificaciones
        self.commands.ee_pose.ranges.pitch = (math.pi/4, math.pi* 3/4)
        self.commands.ee_pose.ranges.yaw = (-math.pi/2, math.pi/2)

        # Contacto con mesa
        self.scene.table.spawn.activate_contact_sensors = True
        self.scene.robot_desk_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            filter_prim_paths_expr = ["{ENV_REGEX_NS}/Robot/*."],
            track_air_time=True,
            track_pose=True,
            update_period=0.0,
            debug_vis=False,
            force_threshold = 1.0
        )

@configclass
class UR3ReachEnvCfg_PLAY(UR3ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


