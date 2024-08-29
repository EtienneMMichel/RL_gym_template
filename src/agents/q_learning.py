import numpy as np
from .utils.agent import Agent


class QlearningAgent:
    def __init__(self, observation_space, action_space, agent_config):
        super().__init__(observation_space, action_space, agent_config)
        self.action_space = action_space
        self.observation_space = observation_space
        self.rewards = []
        self.q_learner = Qlearning(
            learning_rate=agent_config['learner']['learning_rate'],
            gamma=agent_config['learner']['gamma'],
            state_size=self.obs_space_dims,
            action_size=self.action_space_dims)
        self.explorer = EpsilonGreedy(epsilon=agent_config['explorer']['epsilon'])

    def world_obs_2_model_obs(self, obs):
        raise NotImplementedError()

    def model_action_2_world_action(self, action):
        raise NotImplementedError()

    def act(self, obs):
        state = self.world_obs_2_model_obs(obs)
        action = self.explorer.act(
                    action_space=self.action_space, state=state, qtable=self.learner.qtable
                )
        
        return action
    


    def update(self, obs, action, reward, new_obs):
        state = self.world_obs_2_model_obs(obs)
        new_state = self.world_obs_2_model_obs(new_obs)

        self.learner.qtable[state, action] = self.learner.update(
                    state, action, reward, new_state
                )

    def episode_update(self):
        pass

    def save(self, saving_path):
        pass

    def from_save(self, saved_name):
        pass
        

class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def act(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = np.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action