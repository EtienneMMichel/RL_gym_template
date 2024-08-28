import gymnasium as gym


def make_env(env_config):
    env = gym.make('CarRacing-v2')
    return env

def test_env(env_config):
    env = gym.make('CarRacing-v2', render_mode="rgb_array")
    return env