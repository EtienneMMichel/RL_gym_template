import gymnasium as gym


def make_env(env_config):
    env = gym.make(env_config["name"])
    return env

def test_env(env_config):
    env = gym.make(env_config["name"], render_mode="human")
    return env


def generate_envs(env_config):
    return [gym.make(env_config["name"]), gym.make(env_config["name"])]

def init_env(env, seed):
    pass