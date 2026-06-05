from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def posture(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward staying near the default joint pose with a fixed std."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  error_sq = torch.square(joint_pos - default_pos)
  return torch.exp(-torch.mean(error_sq / std**2, dim=1))


def approach_ball(
  env: ManagerBasedRlEnv,
  std: float,
  ball_name: str,
  target_dist: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gaussian reward centered at target_dist from the ball.

  Peaks at 1.0 when dist == target_dist, decays for both closer and farther.
  Set target_dist=0.0 to reward getting as close as possible.
  """
  robot: Entity = env.scene[asset_cfg.name]
  ball: Entity = env.scene[ball_name]

  robot_xy = robot.data.root_link_pos_w[:, :2]
  ball_xy = ball.data.root_link_pos_w[:, :2]
  dist_xy = torch.norm(ball_xy - robot_xy, dim=-1)

  env.extras["log"]["Metrics/ball_distance_mean"] = torch.mean(dist_xy)
  return torch.exp(-((dist_xy - target_dist) ** 2) / (std ** 2))


def face_ball(
  env: ManagerBasedRlEnv,
  ball_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward cosine similarity between robot forward axis and direction to ball.

  Returns +1 when facing straight at the ball, -1 when facing directly away.
  Prevents the backward-walking degenerate strategy.
  """
  robot: Entity = env.scene[asset_cfg.name]
  ball: Entity = env.scene[ball_name]

  to_ball = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  to_ball = to_ball / torch.norm(to_ball, dim=-1, keepdim=True).clamp(min=1e-6)

  forward_w = quat_apply(robot.data.root_link_quat_w, robot.data.forward_vec_b)
  forward_xy = forward_w[:, :2]
  forward_xy = forward_xy / torch.norm(forward_xy, dim=-1, keepdim=True).clamp(min=1e-6)

  return torch.sum(forward_xy * to_ball, dim=-1)
