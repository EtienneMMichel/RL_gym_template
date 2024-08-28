import torch

DEFAULT_GAMMA = 0.99

class ReinforceLossGenerator:
    def __init__(self, loss_config):
        self.gamma = (DEFAULT_GAMMA if not "gamma" in list(loss_config.keys()) else loss_config["gamma"])


    def generate(self):
        def loss_function(rewards, probs):
            running_g = 0
            gs = []

            # Discounted return (backwards) - [::-1] will return an array in reverse
            for R in rewards[::-1]:
                running_g = R + self.gamma * running_g
                gs.insert(0, running_g)

            deltas = torch.tensor(gs)

            loss = 0
            # minimize -1 * prob * reward obtained
            for log_prob, delta in zip(probs, deltas):
                loss += log_prob.mean() * delta * (-1)

            return loss
        return loss_function