from booster_t1_mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import booster_t1_kick_v2_headless_flat_env_cfg
from .rl_cfg import booster_t1_kick_v2_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-AmpKick-Booster-T1-21Dof",
    env_cfg=booster_t1_kick_v2_headless_flat_env_cfg(),
    play_env_cfg=booster_t1_kick_v2_headless_flat_env_cfg(play=True),
    rl_cfg=booster_t1_kick_v2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
