import torch
import random
import numpy as np
from tqdm import tqdm
import yaml
import sys
from src import envs, agents, utils

def train_env(config):
    env = envs.make_env(config["env"])
    is_rendering = config["env"]["render"]

    total_num_episodes = config["total_num_episodes"] #int(5e3)  # Total number of episodes
    seeds =  config["seeds"] # [1, 2, 3, 5, 8]  # Fibonacci seeds
    
    rewards_over_seeds = []
    avg_reward = 0
    infos = {}
    for seed, i_seed in enumerate(seeds):
        infos_in_seed = []
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        agent = eval(f"agents.{config['agent']['name']}(env.observation_space, env.action_space, config['agent'])")
        reward_over_episodes = []
        for episode in tqdm(
            range(total_num_episodes), desc=f"seed {seed} {i_seed}/{len(seeds)} - Episodes - loss {avg_reward}", leave=False
        ):
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

            reward_over_episodes.append(rewards[-1])
            policy_loss = agent.episode_update()
            avg_reward = np.mean(rewards)
            infos_in_seed.append({
                "rewards": rewards,
                "infos": episode_infos,
                "policy_loss": policy_loss,
            })


        rewards_over_seeds.append(reward_over_episodes)
        infos[seed] = infos_in_seed
    
    if config["save"]:
        utils.save(config, agent, infos)


def test_env(config):
    env = envs.test_env(config["env"])
    is_rendering = config["env"]["render"]
    seeds =  config["seeds"]
    rewards_over_seeds = []
    infos = {}
    for seed, i_seed in enumerate(seeds):
        infos_in_seed = []
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        agent = eval(f"agents.{config['agent']['name']}(env.observation_space, env.action_space, config['agent'])")
        reward_over_episodes = []
        state, info = env.reset(seed=seed)
        rewards = []
        episode_infos = []
        done = False
        while not done:
            action = agent.act(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            episode_infos.append(info)
            done = terminated or truncated
            agent.update(state, action, reward, new_state)
            state = new_state
            env.render()

        reward_over_episodes.append(rewards[-1])
        policy_loss = agent.episode_update()
        infos_in_seed.append({
            "rewards": rewards,
            "infos": episode_infos,
            "policy_loss": policy_loss,
        })


        rewards_over_seeds.append(reward_over_episodes)
        infos[seed] = infos_in_seed
    
    if config["save"]:
        utils.save(config, agent, infos)

if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[2], "r"))
    eval(f"{sys.argv[1]}_env(config)")