import gymnasium as gym


def make_env(env_config):
    env = gym.make('CarRacing-v2')
    return env

def test_env(env_config):
    env = gym.make('CarRacing-v2', render_mode="rgb_array")
    return env


def generate_envs(env_config):
    return [gym.make('CarRacing-v2'), gym.make('CarRacing-v2')]

def init_env(env, seed):
    pass