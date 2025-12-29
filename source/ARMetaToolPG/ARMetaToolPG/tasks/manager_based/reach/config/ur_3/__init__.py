
import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="EAS-Reach-UR3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR3ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR3ReachPPORunnerCfg"
    },
)

print("Agente UR3 ha intentado registrarse")