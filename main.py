import torch
import random
import numpy as np
from tqdm import tqdm
import yaml
import sys
from src import envs, agents, utils

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def run_episode(env, agent, seed, is_rendering):
    state, info = env.reset(seed=seed)
    rewards = []
    episode_infos = []
    done = False
    while not done:
        if is_rendering:
            env.render()
        action = agent.act(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        episode_infos.append(info)
        done = terminated or truncated
        agent.update(state, action, reward, new_state)
        state = new_state
    policy_loss = agent.episode_update()
    
    return rewards, episode_infos, policy_loss
        

def train_env(config):
    envs_list = envs.generate_envs(config["env"])
    env = envs_list[0]
    is_rendering = config["env"]["render"]

    total_num_episodes = config["total_num_episodes"] #int(5e3)  # Total number of episodes
    seeds =  config["seeds"] # [1, 2, 3, 5, 8]  # Fibonacci seeds
    
    infos = {}
    for i_seed, seed in enumerate(seeds):
        avg_reward = 0
        infos_in_seed = []
        # set seed
        set_seed(seed)
        
        # Reinitialize agent every seed
        agent = eval(f"agents.{config['agent']['name']}(env.observation_space, env.action_space, config['agent'])")
        for i_env, env in enumerate(envs_list):
            envs.init_env(env, seed)
            for episode in tqdm(
                range(total_num_episodes), desc=f"seed {i_seed}/{len(seeds)} env {i_env}/{len(envs_list)}- Episodes - loss {avg_reward}", leave=False
            ):
                rewards, episode_infos, policy_loss = run_episode(env, agent, seed, is_rendering)
                avg_reward = np.mean(rewards)
                infos_in_seed.append({
                    "rewards": rewards,
                    "infos": episode_infos,
                    "policy_loss": policy_loss,
                })
            env.close()
        infos[seed] = infos_in_seed
    
    if config["save"]:
        utils.save(config, agent, infos)
    
    return infos


def test_env(config):
    env = envs.test_env(config["env"])
    seeds =  config["seeds"]
    infos = {}
    for seed in tqdm(seeds):
        infos_in_seed = []
        set_seed(seed)

        # Reinitialize agent every seed
        agent = eval(f"agents.{config['agent']['name']}(env.observation_space, env.action_space, config['agent'])")
        rewards, episode_infos, policy_loss = run_episode(env, agent, seed, is_rendering=True)
        infos_in_seed.append({
            "rewards": rewards,
            "infos": episode_infos,
            "policy_loss": policy_loss,
        })
        infos[seed] = infos_in_seed

    return infos

if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[2], "r"))
    infos = eval(f"{sys.argv[1]}_env(config)")
    utils.make_summary(infos, sys.argv[1], config["total_num_episodes"])