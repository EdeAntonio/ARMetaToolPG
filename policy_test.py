import os
import torch
import numpy as np

from isaacsim.robot.policy.examples.controllers.config_loader import get_robot_joint_properties, parse_env_config

# Ajusta esta ruta a donde tengas tu modelo
POLICY_FILE = "/home/eantonio/ARMetaToolPG/source/ARMetaToolPG/ARMetaToolPG/assets/policys/policy_pull_object_rh/policy.pt"
POLICY_ENV_FILE= "/home/eantonio/ARMetaToolPG/source/ARMetaToolPG/ARMetaToolPG/assets/policys/policy_pull_object_rh/env.yaml"

print(f"Test de política RoboHabilis: {POLICY_FILE}")

# Verificar que el archivo existe
if not os.path.exists(POLICY_FILE):
    raise FileNotFoundError(f"El archivo de política no existe: {POLICY_FILE}")

# ntentar cargar la política
try:
    policy = torch.jit.load(POLICY_FILE)
    print("Política cargada correctamente.")
except Exception as e:
    print("Error cargando la política:")
    print(e)
    exit(1)

# Probar un forward dummy si la política acepta input
#    Ajusta 'dummy_input' según la entrada que tu política espera
try:
    # Ejemplo: si tu política espera un tensor NxM
    dummy_input = torch.zeros(52)  # Cambia 10 por el tamaño de input de tu policy
    output = policy(dummy_input)
    print("Forward dummy ejecutado correctamente. Salida:")
    print(output)
except Exception as e:
    print("No se pudo ejecutar forward dummy:")
    print(e)
dof_names=[
    "r_shoulder_pan_joint",
    "r_shoulder_lift_joint",
    "r_elbow_joint",
    "r_wrist_1_joint",
    "r_wrist_2_joint",
    "r_wrist_3_joint",
    "r_left_finger_joint",
    "r_right_finger_joint",
    "l_shoulder_pan_joint",
    "l_shoulder_lift_joint",
    "l_elbow_joint",
    "l_wrist_1_joint",
    "l_wrist_2_joint",
    "l_wrist_3_joint",
    "l_left_finger_joint",
    "l_right_finger_joint"
]

policy_env_params = parse_env_config(POLICY_ENV_FILE)

max_effort, max_vel, stiffness, damping, default_pos, default_vel = get_robot_joint_properties(
    policy_env_params, dof_names
)

print(default_pos)
print(default_vel)

print("Test de política finalizado.")
