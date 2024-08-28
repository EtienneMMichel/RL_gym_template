from torch.distributions.normal import Normal


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


class Distribution_Controller():
    def __init__(self, dist_type, action_space_dims) -> None:
        self.dist_type = dist_type
        self.action_space_dims = action_space_dims

    def model_action_2_world_action(self, action_distribution):
        total_log_prob = 0
        total_action = []
        for action_dist_params in action_distribution:
            action, prob = eval(f"{self.dist_type}_sampling(action_dist_params)")
            total_action.append(action)
            total_log_prob += prob
        
        return total_action, total_log_prob