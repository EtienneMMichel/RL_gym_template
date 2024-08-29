import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import os
from functools import reduce
from . import controller


class Gaussian_Perceptron_Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims, action_space_dims, config):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()
        
        self.eps = 1e-6  # small number for mathematical stability
        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change
        inputs_ = reduce((lambda x, y: x * y), obs_space_dims)
        self.controller = controller.Distribution_Controller(dist_type="Gaussian", action_space_dims=action_space_dims)
        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(inputs_, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
            
        )

        self.means_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims),
        )

        self.stddevs_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims),
        )



    def forward(self, x: torch.Tensor):
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        x = torch.flatten(x)
        shared_features = self.shared_net(x.float())
        means = self.means_net(shared_features)
        stddevs = torch.log(
            1 + torch.exp(self.stddevs_net(shared_features))
        )
        action_distribution = []
        for mean, stddev in zip(means, stddevs):
            action_distribution.append({
                "mean": mean,
                "stddev": stddev,
                })

        return self.controller.model_action_2_world_action(action_distribution)


    def save(self, saving_path):
        path = f"{saving_path}/models"
        os.mkdir(path)
        torch.save(self.shared_net.state_dict(), f"{path}/shared_net.pt")
        torch.save(self.means_net.state_dict(), f"{path}/means_net.pt")
        torch.save(self.stddevs_net.state_dict(), f"{path}/stddevs_net.pt")




class Binary_Perceptron_Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims, action_space_dims, config):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()
        
        self.eps = 1e-6  # small number for mathematical stability
        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change
        inputs_ = reduce((lambda x, y: x * y), obs_space_dims)
        self.controller = controller.Distribution_Controller(dist_type="Binary", action_space_dims=action_space_dims)
        self.net = nn.Sequential(
            nn.Linear(inputs_, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
            nn.Linear(hidden_space2, 1),
            nn.Sigmoid()
            
        )


    def forward(self, x: torch.Tensor):
        prob = self.net(x)
        action_distribution = []

        return self.controller.model_action_2_world_action({"p": prob})


    def save(self, saving_path):
        path = f"{saving_path}/models"
        os.mkdir(path)
        torch.save(self.net.state_dict(), f"{path}/net.pt")