import torch
import numpy as np
from .utils.agent import Agent

class Dummy(Agent):

    def __init__(self, observation_space, action_space, agent_config):
        """Initializes an agent
        """
        super().__init__(observation_space, action_space, agent_config)

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        
    def world_obs_2_model_obs(self, obs):
        pass

    def model_action_2_world_action(self, action):
        pass


    def act(self, obs):
        """Returns an action, conditioned on the policy and observation.

        Args:
            obs: Observation from the environment

        Returns:
            action: Action to be performed
        """
        obs = self.world_obs_2_model_obs(obs)
        action = None # where to compute action from model observations
        return self.model_action_2_world_action(action)
    
    def update(self, obs, action, reward, new_obs):
        obs = self.world_obs_2_model_obs(obs)
        new_obs = self.world_obs_2_model_obs(new_obs)
        pass

    def episode_update(self):
        pass

    def save(self, saving_path):
        pass