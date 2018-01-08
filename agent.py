import numpy as np

env_name = "implement a specific game by setting this variable to the env name"

class BaseAgent(object):
    def __init__(self, env):
        self._action_space = range(env.action_space.n)
        self._random = np.random.RandomState(seed = 20171220)

    def select_action(self, observation):
        return self._select_random_action()

    def _select_random_action(self):
        return self._random.choice(self._action_space)

    def learn(self, observation, action, next_observation, reward, done, info):
        pass