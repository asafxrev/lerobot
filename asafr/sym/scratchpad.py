from lerobot.envs.factory import make_env
import numpy as np
import time

# Use the make_env factory to load an HIL environment from the Hugging Face Hub
# The "human" in the repo name indicates support for human input/rendering capabilities
ENV_REPO_ID = "lerobot/act_aloha_sim_insertion_human" 
N_ENVS = 1 # We use 1 environment for interactive human use

# make_env returns a dictionary of vectorized environments; we extract the first one.
# trust_remote_code=True is often required for custom environments from the Hub
envs_dict = make_env(ENV_REPO_ID, n_envs=N_ENVS, trust_remote_code=True)
suite_name = next(iter(envs_dict))
env = envs_dict[suite_name][0]

print(f"Environment loaded: {suite_name}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

try:
    obs, info = env.reset()
    done = np.zeros(env.num_envs, dtype=bool)
    
    while not done.all():
        # In a human-in-the-loop setup, the environment captures human input internally 
        # (e.g., from a keyboard or gamepad) and uses it to override agent actions.
        # A placeholder action (e.g., a random action or a zero action) can be passed
        # if the human is providing all control.
        action = env.action_space.sample() 
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # The environment rendering should be handled automatically by gym-hil
        # or the specific simulation backend when configured correctly (e.g. MuJoCo viewer).

        done = terminated | truncated
        time.sleep(0.01) # Small delay to make the simulation visible if rendering

except KeyboardInterrupt:
    print("Program interrupted by user, closing environment.")
finally:
    env.close()
