from abc import ABC

class VecEnv(ABC):
    """Abstract base class for vectorized environments."""
    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buf: object
    obs_buf: object
    rew_buf: object
    reset_buf: object
    episode_length_buf: object
    extras: dict
    device: str
