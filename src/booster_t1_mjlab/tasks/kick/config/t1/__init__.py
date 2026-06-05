from booster_t1_mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import booster_t1_kick_flat_env_cfg, booster_t1_kick_headless_flat_env_cfg
from .rl_cfg import booster_t1_kick_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Kick-Flat-Booster-T1",
  env_cfg=booster_t1_kick_flat_env_cfg(),
  play_env_cfg=booster_t1_kick_flat_env_cfg(play=True),
  rl_cfg=booster_t1_kick_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

_headless_rl_cfg = booster_t1_kick_ppo_runner_cfg()
_headless_rl_cfg.experiment_name = "t1_kick_headless"

register_mjlab_task(
  task_id="Mjlab-Kick-Flat-Booster-T1-Headless",
  env_cfg=booster_t1_kick_headless_flat_env_cfg(),
  play_env_cfg=booster_t1_kick_headless_flat_env_cfg(play=True),
  rl_cfg=_headless_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)
