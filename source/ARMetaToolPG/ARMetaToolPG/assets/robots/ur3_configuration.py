

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from ARMetaToolPG.assets import ARMT_ASSETS_DATA_DIR

UR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ARMT_ASSETS_DATA_DIR}/ur3e/ur3e_gripper.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)