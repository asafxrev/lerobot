from lerobot.envs.factory import make_env
import numpy as np

# Load the environment
envs_dict = make_env("lerobot/cartpole-env", n_envs=4, trust_remote_code=True)

# Get the vectorized environment
suite_name = next(iter(envs_dict))
env = envs_dict[suite_name][0]

# Run a simple episode
obs, info = env.reset()
env.render_mode = "human"
done = np.zeros(env.num_envs, dtype=bool)
total_reward = np.zeros(env.num_envs)

while not done.all():
    # Random policy
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated | truncated

print(f"Average reward: {total_reward.mean():.2f}")
env.close()