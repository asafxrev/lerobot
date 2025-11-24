from lerobot.envs.utils import _load_module_from_path, _call_make_env, _normalize_hub_result
import numpy as np
import time

# Load your module
module = _load_module_from_path("./inverted_pendulum_env.py")

# Test the make_env function
result = _call_make_env(module, n_envs=2, use_async_envs=False)
normalized = _normalize_hub_result(result)

# Verify it works
suite_name = next(iter(normalized))
env = normalized[suite_name][0]
obs, info = env.reset()

# Run a simple episode
obs, info = env.reset()
env.render_mode = "human"
done = np.zeros(env.num_envs, dtype=bool)
total_reward = np.zeros(env.num_envs)


def get_user_actions() -> list[np.ndarray]|None:
    # TODO: Update. The current code is for cartpole.
    action = None
    while action is None:
        inp = input('Choose action and repetition. '
                    'Left (L) or right (R), followed by number of repetitions (e.g., "L3" for left 3 times). '
                    '-1 to quit: ')
        if inp == '-1':
            return None
        
        if len(inp) < 1 or not inp[0].lower() in ['l', 'r']:
            print('Invalid input. Please enter a direction (L or R) followed by number of repetitions. - 1 to quit.')
            continue

        direction = {
            'l': np.array([1]),
            'r': np.array([0]),
        }[inp[0].lower()]
        try:
            repetitions = int(inp[1:]) if len(inp) > 1 else 1
        except ValueError:
            print('Invalid number of repetitions. Please enter a valid integer after the direction.')
            continue

        return [direction] * repetitions



print('''
Inverted Pendulum      
https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/
      
''')


user_actions = []
while not done.all():
    # Random policy
    time.sleep(0.3)  # Small delay for better visualization
    action = env.action_space.sample()

    # # Interactive action selection
    # if len(user_actions) == 0:
    #     user_actions = get_user_actions()
    #     if user_actions is None:
    #         print('Exiting the episode.')
    #         break
    # action = user_actions.pop(0)
    # print(f'Executing action: {action}')

    obs, reward, terminated, truncated, info = env.step(action)
    print(f'Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}')
    total_reward += reward
    done = terminated | truncated

print(f"Average reward: {total_reward.mean():.2f}")
env.close()