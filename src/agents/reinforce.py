import torch
import numpy as np
from .utils.deep_agent import DeepAgent

class REINFORCE(DeepAgent):
    """REINFORCE algorithm."""

    def __init__(self, observation_space, action_space, agent_config):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
            policy_model_name: Poly model name to train
        """
        super().__init__(observation_space, action_space, agent_config)
        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        

        
        
    def world_obs_2_model_obs(self, obs):
        # May include VAE model to exploit latent space
        return torch.tensor(np.array([obs]))

    


    def act(self, obs):
        """Returns an action, conditioned on the policy and observation.

        Args:
            obs: Observation from the environment

        Returns:
            action: Action to be performed
        """
        obs = self.world_obs_2_model_obs(obs)
        action, prob = self.policy(obs)
        self.probs.append(prob)
        return action

    
    def update(self, obs, action, reward, new_obs):
        self.rewards.append(reward)

    def episode_update(self):
        """Updates the policy network's weights."""
        loss = self.loss(rewards=self.rewards, probs=self.probs)
        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

        return loss.detach().numpy().tolist()
    
    