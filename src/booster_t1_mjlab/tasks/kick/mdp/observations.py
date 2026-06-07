from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))


def ball_pos_xy_robot_frame(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Ball XY offset relative to foot midpoint, expressed in robot body frame.

    Origin: midpoint between left_foot_link and right_foot_link.
    Orientation: robot base yaw frame (x=forward, y=left).
    Returns [x_forward, y_left] — no height component.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    foot_pos = robot.data.body_link_pos_w[:, feet_asset_cfg.body_ids, :2]  # [B, 2, 2]
    feet_mid_xy = foot_pos.mean(dim=1)  # [B, 2]

    ball_xy = ball.data.root_link_pos_w[:, :2]  # [B, 2]

    rel_w = torch.zeros(feet_mid_xy.shape[0], 3, device=feet_mid_xy.device)
    rel_w[:, :2] = ball_xy - feet_mid_xy

    rel_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_w)  # [B, 3]
    return rel_b[:, :2]  # [B, 2]: [x_forward, y_left]


def kick_shot_angle_obs(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    """Commanded kick direction in robot frame, wrapped to [-π, π]. Shape [N, 1].

    Zero once the kick is in flight (_kick_timer > 0) — the ball direction is
    already committed at that point, so the policy should stop trying to steer.
    """
    if not hasattr(env, "_kick_world_shot_angle"):
        return torch.zeros(env.num_envs, 1, device=env.device)

    q = env.scene["robot"].data.root_link_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    delta = env._kick_world_shot_angle - robot_yaw
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    # freeze to zero once kick is detected — direction can't change after impact
    delta[env._kick_timer > 0] = 0.0
    return delta.unsqueeze(-1)


def kick_target_speed_obs(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    """Commanded ball speed after kick [m/s]. Shape [N, 1]."""
    if not hasattr(env, "_kick_target_speed"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env._kick_target_speed.unsqueeze(-1)
