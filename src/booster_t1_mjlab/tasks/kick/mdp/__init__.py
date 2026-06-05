"""MDP components for the kick task."""

from booster_t1_mjlab.tasks.kick.mdp.events import reset_ball_ahead_of_origin
from booster_t1_mjlab.tasks.kick.mdp.observations import ball_pos_xy_robot_frame
from booster_t1_mjlab.tasks.kick.mdp.rewards import approach_ball, face_ball, posture
