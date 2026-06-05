"""Booster T1 ball-approach environment configuration."""

from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from booster_t1_mjlab.tasks.kick import mdp as kick_mdp
from booster_t1_mjlab.robots import get_t1_headless_robot_cfg, T1_ACTION_SCALE_HEADLESS
from booster_t1_mjlab.tasks.velocity.config.t1.env_cfgs import booster_t1_flat_env_cfg

BALL_XML: Path = Path(__file__).parents[4] / "robots" / "boostert1" / "xmls" / "ball.xml"
assert BALL_XML.exists(), f"ball.xml not found at {BALL_XML}"

BALL_NAME = "ball"

_ALL_JOINTS_CFG = SceneEntityCfg("robot", joint_names=(".*",))
_FEET_ASSET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))


def _make_ball_entity_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file(str(BALL_XML)),
    init_state=EntityCfg.InitialStateCfg(
      pos=(1.0, 0.0, 0.11),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={},
    ),
  )


def booster_t1_kick_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """T1 ball-approach task: robot spawns facing the ball and walks to it.

  The only ball-specific observation is the 2-D offset [x_forward, y_left]
  in the robot body frame — no height component. The only ball-specific reward
  is approach_ball (exponential in XY distance).
  """
  cfg = booster_t1_flat_env_cfg(play=play)

  # -------------------------------------------------------------------------
  # Scene: add the ball entity
  # -------------------------------------------------------------------------
  cfg.scene.entities[BALL_NAME] = _make_ball_entity_cfg()

  # -------------------------------------------------------------------------
  # Commands: remove the velocity twist command
  # -------------------------------------------------------------------------
  cfg.commands.clear()

  # -------------------------------------------------------------------------
  # Curriculum: remove command-dependent curriculum
  # -------------------------------------------------------------------------
  cfg.curriculum.clear()

  # -------------------------------------------------------------------------
  # Observations: replace "command" with ball XY in robot body frame
  # -------------------------------------------------------------------------
  ball_pos_noisy = ObservationTermCfg(
    func=kick_mdp.ball_pos_xy_robot_frame,
    params={"ball_name": BALL_NAME, "feet_asset_cfg": _FEET_ASSET_CFG},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )
  ball_pos_clean = ObservationTermCfg(
    func=kick_mdp.ball_pos_xy_robot_frame,
    params={"ball_name": BALL_NAME, "feet_asset_cfg": _FEET_ASSET_CFG},
  )

  del cfg.observations["actor"].terms["command"]
  cfg.observations["actor"].terms["ball_pos"] = ball_pos_noisy

  del cfg.observations["critic"].terms["command"]
  cfg.observations["critic"].terms["ball_pos"] = ball_pos_clean

  # -------------------------------------------------------------------------
  # Rewards
  # -------------------------------------------------------------------------
  del cfg.rewards["track_linear_velocity"]
  del cfg.rewards["track_angular_velocity"]

  cfg.rewards["action_rate_l2"].weight = -0.5

  del cfg.rewards["foot_slip"]
  del cfg.rewards["foot_swing_height"]

  del cfg.rewards["pose"]
  cfg.rewards["pose"] = RewardTermCfg(
    func=kick_mdp.posture,
    weight=1.0,
    params={"std": 0.25, "asset_cfg": _ALL_JOINTS_CFG},
  )

  cfg.rewards["air_time"].params.pop("command_name", None)
  cfg.rewards["air_time"].params.pop("command_threshold", None)
  cfg.rewards["foot_clearance"].params.pop("command_name", None)
  cfg.rewards["foot_clearance"].params.pop("command_threshold", None)
  cfg.rewards["soft_landing"].params.pop("command_name", None)
  cfg.rewards["soft_landing"].params.pop("command_threshold", None)

  cfg.rewards["approach_ball"] = RewardTermCfg(
    func=kick_mdp.approach_ball,
    weight=2.0,
    params={"std": 0.15, "ball_name": BALL_NAME, "target_dist": 0.6},
  )
  cfg.rewards["face_ball"] = RewardTermCfg(
    func=kick_mdp.face_ball,
    weight=1.0,
    params={"ball_name": BALL_NAME},
  )

  # -------------------------------------------------------------------------
  # Events
  # -------------------------------------------------------------------------
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (-0.15, 0.15)

  cfg.events["reset_ball"] = EventTermCfg(
    func=kick_mdp.reset_ball_ahead_of_origin,
    mode="reset",
    params={
      "ball_name": BALL_NAME,
      "distance_range": (0.8, 2.0),
      "y_range": (-1.0, 1.0),
      "ball_radius": 0.11,
    },
  )

  # -------------------------------------------------------------------------
  # Play-mode overrides
  # -------------------------------------------------------------------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg


def booster_t1_kick_headless_flat_env_cfg(play: bool = False):
  """Kick task with head joints fixed — 21-DOF policy (no head in obs/action)."""
  cfg = booster_t1_kick_flat_env_cfg(play=play)
  cfg.scene.entities["robot"] = get_t1_headless_robot_cfg()
  cfg.actions["joint_pos"].scale = T1_ACTION_SCALE_HEADLESS
  return cfg
