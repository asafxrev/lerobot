from lerobot.envs.utils import _load_module_from_path, _call_make_env, _normalize_hub_result
import numpy as np

# Load your module
module = _load_module_from_path("./sym_env.py")

# Test the make_env function
result = _call_make_env(module, n_envs=1, use_async_envs=False)
normalized = _normalize_hub_result(result)

# Verify it works
suite_name = next(iter(normalized))
env = normalized[suite_name][0]
obs, info = env.reset()
# print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
# env.close()

# Run a simple episode
obs, info = env.reset()
env.render_mode = "human"
done = np.zeros(env.num_envs, dtype=bool)
total_reward = np.zeros(env.num_envs)


def get_user_actions() -> list[np.ndarray]|None:
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
CartPole Control
      
https://gymnasium.farama.org/environments/classic_control/cart_pole/
      
Obs is [cart position, cart velocity, pole angle, pole angular velocity]

* The cart x-position (index 0) can be take values between (-4.8, 4.8), but the episode terminates if the cart leaves the (-2.4, 2.4) range.
* The pole angle can be observed between (-.418, .418) radians (or ±24°), but the episode terminates if the pole angle is not in the range (-.2095, .2095) (or ±12°)
''')


user_actions = []
while not done.all():
    # Random policy
    # action = env.action_space.sample()

    # Interactive action selection
    if len(user_actions) == 0:
        user_actions = get_user_actions()
        if user_actions is None:
            print('Exiting the episode.')
            break
    action = user_actions.pop(0)
    print(f'Executing action: {action}')

    obs, reward, terminated, truncated, info = env.step(action)
    print(f'Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}')
    total_reward += reward
    done = terminated | truncated

print(f"Average reward: {total_reward.mean():.2f}")
env.close()