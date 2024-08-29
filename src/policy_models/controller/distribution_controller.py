from torch.distributions.normal import Normal
import torch
import numpy as np

def Gaussian_sampling(action_dist_params):
    '''
    action_dist_params : {"mean":float, "stddev":float}
    '''
    eps = 1e-6
    distrib = Normal(action_dist_params["mean"] + eps, action_dist_params["stddev"] + eps)
    action = distrib.sample()
    # action = (torch.tensor(1) if action > .5 else torch.tensor(0)) # IF SCALAR (Discrete)
    prob = distrib.log_prob(action)
    action = action.numpy().tolist()
    return action, prob


def Binary_sampling(action_dist_params):
    '''
    action_dist_params : {"p":float}
    p: probability to choose action 0
    '''
    p = action_dist_params["p"][0][0].clone().detach().numpy()
    action = np.random.choice([0,1], p=[p, 1 - p])
    prob = (action_dist_params["p"][0][0] if action == 0 else torch.tensor(1)-action_dist_params["p"][0][0])
    # action = torch.tensor(action)
    return action, prob

class Distribution_Controller():
    def __init__(self, dist_type, action_space_dims) -> None:
        self.dist_type = dist_type
        self.action_space_dims = action_space_dims

    def model_action_2_world_action(self, action_distribution):
        if isinstance(action_distribution, list):
            total_log_prob = 0
            total_action = []
            for action_dist_params in action_distribution:
                action, prob = eval(f"{self.dist_type}_sampling(action_dist_params)")
                total_action.append(action)
                total_log_prob += prob
        elif isinstance(action_distribution, dict):
            # Discrete case
            total_action, total_log_prob = eval(f"{self.dist_type}_sampling(action_distribution)")
        
        return total_action, total_log_prob