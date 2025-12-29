import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
# from MT_ext.assets.robots import MT_ASSETS_DATA_DIR
from ARMetaToolPG.assets import ARMT_ASSETS_DATA_DIR


L_ROBOHABILIS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ARMT_ASSETS_DATA_DIR}/robohabilis/L_robohabilis_Isaaclab.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # left arm
            "l_shoulder_pan_joint": -3.86,
            "l_shoulder_lift_joint": 0.3,
            "l_elbow_joint": -2.24,
            "l_wrist_1_joint": 0.91,
            "l_wrist_2_joint": 2.22,
            "l_wrist_3_joint": -1.75,
            "l_left_finger_joint": 0.0,
            "l_right_finger_joint": 0.0,
        },
    ),
    actuators={
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["l_shoulder_pan_joint", "l_shoulder_lift_joint", "l_elbow_joint", 
                              "l_wrist_1_joint", "l_wrist_2_joint", "l_wrist_3_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=210.0,
            damping=21.0,
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["l_.*_finger_joint"],
            velocity_limit=200.0,
            effort_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)

ROBOHABILIS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ARMT_ASSETS_DATA_DIR}/robohabilis/robohabilis_Isaaclab.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # right arm
            "r_shoulder_pan_joint": 3.50,
            "r_shoulder_lift_joint": 0.25,
            "r_elbow_joint": -0.8,
            "r_wrist_1_joint": 0.2,
            "r_wrist_2_joint": 0.9,
            "r_wrist_3_joint": 0.0,
            "r_left_finger_joint": 0.0,
            "r_right_finger_joint": 0.0,
            # left arm
            "l_shoulder_pan_joint": -3.86,
            "l_shoulder_lift_joint": 0.3,
            "l_elbow_joint": -2.24,
            "l_wrist_1_joint": 0.91,
            "l_wrist_2_joint": 2.22,
            "l_wrist_3_joint": -1.75,
            "l_left_finger_joint": 0.0,
            "l_right_finger_joint": 0.0,
        },
    ),
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["r_shoulder_pan_joint", "r_shoulder_lift_joint", "r_elbow_joint", 
                              "r_wrist_1_joint", "r_wrist_2_joint", "r_wrist_3_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=210.0,
            damping=21.0,
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["r_.*_finger_joint"],
            velocity_limit=200.0,
            effort_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["l_shoulder_pan_joint", "l_shoulder_lift_joint", "l_elbow_joint", 
                              "l_wrist_1_joint", "l_wrist_2_joint", "l_wrist_3_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=210.0,
            damping=21.0,
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["l_.*_finger_joint"],
            velocity_limit=200.0,
            effort_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)