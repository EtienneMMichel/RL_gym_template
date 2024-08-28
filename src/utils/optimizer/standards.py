import torch

DEFAULT_LEARNING_RATE = 0.01

class AdamWGenerator:
    def __init__(self, loss_config):
        self.learning_rate = (DEFAULT_LEARNING_RATE if not "learning_rate" in list(loss_config.keys()) else loss_config["learning_rate"])

    def generate(self, net):
        return torch.optim.AdamW(net.parameters(), lr=self.learning_rate)