"""MDP components for the kick task."""

from booster_t1_mjlab.tasks.kick.mdp.events import (
    kick_cycle_step,
    reset_ball_ahead_of_origin,
    reset_kick_state,
)
from booster_t1_mjlab.tasks.kick.mdp.observations import (
    ball_pos_xy_robot_frame,
    kick_shot_angle_obs,
    kick_target_speed_obs,
)
from booster_t1_mjlab.tasks.kick.mdp.rewards import (
    approach_ball,
    face_ball,
    face_shot_direction,
    kick_direction,
    kick_speed,
    posture,
)
