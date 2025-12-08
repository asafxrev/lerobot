# See https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/

import gymnasium as gym
import env

def make_env(n_envs: int = 1, use_async_envs: bool = False):
    """
    Create vectorized environments for your custom task.

    Args:
        n_envs: Number of parallel environments
        use_async_envs: Whether to use AsyncVectorEnv or SyncVectorEnv

    Returns:
        gym.vector.VectorEnv or dict mapping suite names to vectorized envs
    """
    return env.make_env_detailed(
        n_envs, use_async_envs, render_mode=None)