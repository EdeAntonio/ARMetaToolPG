
from __future__ import annotations
from isaaclab.utils.math import quat_apply, quat_inv, euler_xyz_from_quat, quat_mul
import torch

def generated_commands_base_frame(env, command_name: str | None = None) -> torch.Tensor: 
    
    # Obtener la comanda
    cmd = env.command_manager.get_command(command_name)
    pos_world = cmd[:, :3]
    quat_world= cmd[:, 3:7]

    base_pos_world = env.scene["robot"].data.root_state_w[:, :3]
    base_quat_world = env.scene["robot"].data.root_state_w[:, 3:7]

    rel_pos = torch.sub(pos_world, base_pos_world)
    rel_pos = quat_apply(quat_inv(base_quat_world), rel_pos)


    quat_rel = quat_mul(quat_inv(base_quat_world), quat_world)
    roll, pitch, yaw = euler_xyz_from_quat(quat_rel)
    rpy_rel = torch.stack([roll, pitch, yaw], dim=-1)

    return torch.cat([rel_pos, rpy_rel], dim=-1)
        