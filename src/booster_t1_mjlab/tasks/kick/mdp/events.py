from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Shared kick-state helpers
# ---------------------------------------------------------------------------

def _ensure_kick_state(env: "ManagerBasedRlEnv") -> None:
    """Lazily allocate per-env kick-state tensors on the env object."""
    if not hasattr(env, "_kick_timer"):
        N = env.num_envs
        env._kick_timer = torch.zeros(N, dtype=torch.long, device=env.device)
        env._kick_world_shot_angle = torch.zeros(N, device=env.device)
        env._kick_target_speed = torch.zeros(N, device=env.device)
        env._kick_ball_vel_at_kick = torch.zeros(N, 3, device=env.device)


def _robot_yaw(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Yaw angle [N] extracted from robot root quaternion (wxyz)."""
    q = env.scene["robot"].data.root_link_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------------------
# Existing event (unchanged)
# ---------------------------------------------------------------------------

def _ball_pos_ahead_of_robot(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    distance_range: tuple[float, float],
    y_range: tuple[float, float],
    ball_radius: float,
) -> torch.Tensor:
    """Return ball XYZ positions placed in front of the robot's current pose."""
    robot = env.scene["robot"]
    robot_pos = robot.data.root_link_pos_w[env_ids]  # [N, 3]
    yaw = _robot_yaw(env)[env_ids]                   # [N]

    n = len(env_ids)
    dist  = sample_uniform(*distance_range, (n,), env.device)
    y_off = sample_uniform(*y_range, (n,), env.device)

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    ball_x = robot_pos[:, 0] + dist * cos_yaw - y_off * sin_yaw
    ball_y = robot_pos[:, 1] + dist * sin_yaw + y_off * cos_yaw
    ball_z = env.scene.env_origins[env_ids, 2] + ball_radius

    return torch.stack([ball_x, ball_y, ball_z], dim=-1)


def reset_ball_ahead_of_origin(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    distance_range: tuple[float, float] = (0.8, 2.0),
    y_range: tuple[float, float] = (0.0, 0.0),
    ball_radius: float = 0.11,
) -> None:
    """Reset the ball in front of the robot's actual spawn pose."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    ball: Entity = env.scene[ball_name]
    N = len(env_ids)

    ball_pos = _ball_pos_ahead_of_robot(env, env_ids, distance_range, y_range, ball_radius)
    ball_quat = torch.zeros((N, 4), device=env.device)
    ball_quat[:, 0] = 1.0
    ball_vel = torch.zeros((N, 6), device=env.device)

    ball.write_root_state_to_sim(
        torch.cat([ball_pos, ball_quat, ball_vel], dim=-1),
        env_ids=env_ids,
    )


# ---------------------------------------------------------------------------
# New events for v2 kick task
# ---------------------------------------------------------------------------

def reset_kick_state(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    shot_angle_offset_range: tuple[float, float] = (-math.pi / 3, math.pi / 3),
    target_speed_range: tuple[float, float] = (2.0, 8.0),
) -> None:
    """Sample kick command (world shot angle + target ball speed) at episode reset."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    _ensure_kick_state(env)
    n = len(env_ids)

    offset = sample_uniform(*shot_angle_offset_range, (n,), env.device)
    env._kick_world_shot_angle[env_ids] = _robot_yaw(env)[env_ids] + offset
    env._kick_target_speed[env_ids] = sample_uniform(*target_speed_range, (n,), env.device)
    env._kick_timer[env_ids] = 0
    env._kick_ball_vel_at_kick[env_ids] = 0.0


def kick_cycle_step(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,  # all envs (interval_range=(1,1))
    ball_name: str = "ball",
    speed_threshold: float = 1.5,
    reset_delay_steps: int = 10,
    ball_reset_prob: float = 0.9,
    distance_range: tuple[float, float] = (0.2, 0.6),
    y_range: tuple[float, float] = (-0.3, 0.3),
    ball_radius: float = 0.11,
    shot_angle_offset_range: tuple[float, float] = (-math.pi / 3, math.pi / 3),
    target_speed_range: tuple[float, float] = (2.0, 8.0),
) -> None:
    """Per-step kick-cycle manager.

    1. Detect when the ball crosses ``speed_threshold`` m/s → record its
       velocity and start a 200 ms (``reset_delay_steps`` steps) timer.
    2. After the timer expires:
       - With probability ``ball_reset_prob`` (90 %): move the ball back in
         front of the robot and zero its velocity (new kick attempt).
       - Otherwise (10 %): leave the ball where it rolled so the robot must
         chase and re-kick it.
    3. Always: sample a fresh shot_angle + target_speed for all expired envs.
    """
    _ensure_kick_state(env)
    ball: Entity = env.scene[ball_name]

    ball_lin_vel = ball.data.root_link_lin_vel_w  # [N, 3]
    ball_speed = torch.norm(ball_lin_vel, dim=-1)  # [N]

    # -- start timer on first frame ball exceeds threshold --
    just_kicked = (ball_speed > speed_threshold) & (env._kick_timer == 0)
    if just_kicked.any():
        env._kick_ball_vel_at_kick[just_kicked] = ball_lin_vel[just_kicked].clone()
        env._kick_timer[just_kicked] = 1

    # -- advance running timers --
    env._kick_timer[env._kick_timer > 0] += 1

    # -- handle expired timers --
    expired_mask = env._kick_timer > reset_delay_steps
    if not expired_mask.any():
        return

    expired_ids = expired_mask.nonzero(as_tuple=False).squeeze(-1)
    n_exp = len(expired_ids)

    # decide which get a full ball reset
    do_full = torch.rand(n_exp, device=env.device) < ball_reset_prob
    full_ids = expired_ids[do_full]

    if len(full_ids) > 0:
        n_full = len(full_ids)
        ball_pos = _ball_pos_ahead_of_robot(env, full_ids, distance_range, y_range, ball_radius)
        ball_quat = torch.zeros((n_full, 4), device=env.device)
        ball_quat[:, 0] = 1.0
        ball_vel = torch.zeros((n_full, 6), device=env.device)
        ball.write_root_state_to_sim(
            torch.cat([ball_pos, ball_quat, ball_vel], dim=-1),
            env_ids=full_ids,
        )

    # resample kick command for all expired envs
    offset = sample_uniform(*shot_angle_offset_range, (n_exp,), env.device)
    env._kick_world_shot_angle[expired_ids] = _robot_yaw(env)[expired_ids] + offset
    env._kick_target_speed[expired_ids] = sample_uniform(*target_speed_range, (n_exp,), env.device)
    env._kick_timer[expired_ids] = 0
    env._kick_ball_vel_at_kick[expired_ids] = 0.0
