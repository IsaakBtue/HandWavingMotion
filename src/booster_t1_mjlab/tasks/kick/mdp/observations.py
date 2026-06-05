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

_step = 0


def ball_pos_xy_robot_frame(
  env: ManagerBasedRlEnv,
  ball_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  feet_asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
  """Ball XY offset relative to foot midpoint, expressed in robot body frame.

  Origin: midpoint between left_foot_link and right_foot_link (matches real
  robot software convention). Orientation: robot base yaw frame (x=forward,
  y=left). Returns [x_forward, y_left] with no height component.
  """
  robot: Entity = env.scene[asset_cfg.name]
  ball: Entity = env.scene[ball_name]

  foot_pos = robot.data.body_link_pos_w[:, feet_asset_cfg.body_ids, :2]  # [B, 2, 2]
  feet_mid_xy = foot_pos.mean(dim=1)  # [B, 2]

  ball_xy = ball.data.root_link_pos_w[:, :2]  # [B, 2]

  # Lift to 3D with z=0 so quat_apply_inverse works
  rel_w = torch.zeros(feet_mid_xy.shape[0], 3, device=feet_mid_xy.device)
  rel_w[:, :2] = ball_xy - feet_mid_xy

  rel_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_w)  # [B, 3]

  out = rel_b[:, :2]  # [B, 2]: [x_forward, y_left]

  global _step
  _step += 1
  if _step % 50 == 0:
    print(f"[ball_pos] x_fwd={out[0, 0]:.3f}  y_left={out[0, 1]:.3f}")

  return out
