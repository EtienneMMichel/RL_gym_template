from ... import policy_models
from ... import utils
from .agent import Agent

class DeepAgent(Agent):

    def __init__(self, observation_space, action_space, agent_config):
        super().__init__(observation_space, action_space, agent_config)
        self.policy = self.build_policy(agent_config["policy"])
        self.loss = self.build_loss(agent_config["loss"])
        self.optimizer = self.build_optimizer(agent_config["optimizer"])

    def build_policy(self, policy_config):
        return eval(f"policy_models.{policy_config['name']}(self.obs_space_dims, self.action_space_dims, policy_config)")

    def build_optimizer(self, optimizer_config):
        optimizer_generator = eval(f"utils.optimizer.{optimizer_config['name']}(optimizer_config)")
        return optimizer_generator.generate(self.policy)

    
    def build_loss(self, loss_config):
        loss_generator = eval(f"utils.loss.{loss_config['name']}(loss_config)")
        return loss_generator.generate()
    
    def save(self, saving_path):
        self.policy.save(saving_path)
        

    