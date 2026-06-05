from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ball_kicked(
  env: ManagerBasedRlEnv,
  ball_name: str,
  speed_threshold: float = 0.5,
) -> torch.Tensor:
  """Terminate when the ball exceeds speed_threshold m/s.

  Ends the episode as soon as the robot kicks the ball so it only needs to
  kick once per episode. The robot then resets and tries again from scratch.
  """
  ball: Entity = env.scene[ball_name]
  ball_speed = torch.norm(ball.data.root_link_lin_vel_w, dim=-1)  # [B]
  return ball_speed > speed_threshold
