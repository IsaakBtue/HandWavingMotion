"""Kick AMP environment config (beyondAMP backend)."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from beyondAMP.mjlab.obs_groups import amp_obs_basic_group

from booster_t1_mjlab.tasks.kick.config.t1.env_cfgs import (
    booster_t1_kick_v2_headless_flat_env_cfg,
)

KICK_ANCHOR_NAME: str = "Trunk"

KICK_KEY_BODY_NAMES: list[str] = [
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
    "Waist",
]


def t1_amp_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = booster_t1_kick_v2_headless_flat_env_cfg(play=play)
    cfg.observations["amp"] = amp_obs_basic_group()
    return cfg
