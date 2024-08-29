import torch.nn as nn
import torch
import numpy as np

class Dummy(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims, action_space_dims, config):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()
        self.obs_space_dims = obs_space_dims
        self.action_space_dims = action_space_dims


    def forward(self, obs: torch.Tensor):
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            
        """
        res = np.random.random(self.action_space_dims)

        return res
    
    def save(self, saving_path):
        pass

    def from_save(self, saved_name):
        pass