# See https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/

import gymnasium as gym

def make_env(n_envs: int = 1, use_async_envs: bool = False):
    return make_env_detailed(n_envs, use_async_envs, render_mode='human')


def make_env_detailed(n_envs: int = 1, use_async_envs: bool = False, 
                      render_mode: str = 'human'):
    """
    Create vectorized environments for your custom task.

    Args:
        n_envs: Number of parallel environments
        use_async_envs: Whether to use AsyncVectorEnv or SyncVectorEnv

    Returns:
        gym.vector.VectorEnv or dict mapping suite names to vectorized envs
    """
    def _make_single_env():
        # Create your custom environment
        return gym.make('InvertedDoublePendulum-v4', 
                        # reset_noise_scale=0.1, 
                        render_mode=render_mode,
                        width=1280, height=720)

    # Choose vector environment type
    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    # Create vectorized environment
    vec_env = env_cls([_make_single_env for _ in range(n_envs)])

    return vec_env