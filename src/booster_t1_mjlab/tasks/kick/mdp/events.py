from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def reset_ball_ahead_of_origin(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ball_name: str,
  distance_range: tuple[float, float] = (0.8, 2.0),
  y_range: tuple[float, float] = (0.0, 0.0),
  ball_radius: float = 0.11,
) -> None:
  """Reset the ball directly ahead of the env origin in the world +x direction.

  Reading robot.data.root_link_quat_w after reset_base writes new qpos is
  unreliable — xquat isn't refreshed until the next mj_forward(). Using
  env_origins (which never change) sidesteps the stale-state issue entirely.

  Pair with a robot reset that constrains yaw to a small range so the robot
  starts facing roughly +x.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  ball: Entity = env.scene[ball_name]
  N = len(env_ids)

  origins = env.scene.env_origins[env_ids]  # [N, 3]
  dist = sample_uniform(*distance_range, (N,), env.device)

  ball_x = origins[:, 0] + dist
  ball_y = origins[:, 1] + sample_uniform(*y_range, (N,), env.device)
  ball_z = origins[:, 2] + ball_radius

  ball_pos = torch.stack([ball_x, ball_y, ball_z], dim=-1)
  ball_quat = torch.zeros((N, 4), device=env.device)
  ball_quat[:, 0] = 1.0  # identity [w, x, y, z]
  ball_vel = torch.zeros((N, 6), device=env.device)

  ball_state = torch.cat([ball_pos, ball_quat, ball_vel], dim=-1)  # [N, 13]
  ball.write_root_state_to_sim(ball_state, env_ids=env_ids)
