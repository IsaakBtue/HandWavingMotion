"""Register T1 AMP kick task in mjlab.tasks.registry (beyondAMP runner).

Train with:
    uv run booster_t1_train_beyondamp Mjlab-AmpKick-Booster-T1-21Dof --num-envs 4096
"""

from mjlab.tasks.registry import register_mjlab_task

from .amp_ppo_cfg import t1_amp_kick_runner_cfg
from ..amp_env_cfg import t1_amp_kick_env_cfg

register_mjlab_task(
    task_id="Mjlab-AmpKick-Booster-T1-21Dof",
    env_cfg=t1_amp_kick_env_cfg(),
    play_env_cfg=t1_amp_kick_env_cfg(play=True),
    rl_cfg=t1_amp_kick_runner_cfg(),
    runner_cls=None,
)
