from agent import BaseAgent

env_name = "MountainCar-v0"

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from random import sample

class Agent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)
        self._state_size = env.observation_space.shape[0]
        self._action_size = env.action_space.n
        self._gamma = 0.95    # discount rate
        self._epsilon = 1.0  # exploration rate
        self._epsilon_min = 0.01
        self._epsilon_decay = 0.99996
        self._model = self._build_model()
        self._model_target = self._build_model()
        self._memory = deque(maxlen=20000)
        self._learning_batch_size = 25
        self._learning_steps_delay = 200

    def _update_target_network(self):
        self._model_target.set_weights(self._model.get_weights())

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self._state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        optimizer = Adam(lr=0.00025)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def select_action(self, observation):
        if self._random.rand() <= self._epsilon:
            return self._select_random_action()
        observation = self._transform_observation(observation)
        act_values = self._model.predict(observation)
        return np.argmax(act_values[0])

    def _transform_observation(self, observation):
        transformed = np.reshape(observation, [1, self._state_size])
        return transformed

    def learn(self, observation, action, next_observation, reward, done, info):
        self._memory.append((
            observation, 
            action, 
            next_observation, 
            reward, 
            done, 
            info
        ))
        if (len(self._memory) > self._learning_steps_delay):
            self._learn_from_memory()
            self._decay_epsilon()

    def _learn_from_memory(self):
        batch = np.array(sample(self._memory, self._learning_batch_size))
        observations = np.array(batch[:,0].tolist())
        actions = np.array(batch[:,1].tolist())
        observations_next = np.array(batch[:,2].tolist())
        rewards = np.array(batch[:,3].tolist())
        dones = np.array(batch[:,4].tolist())
        Qvalues = self._model.predict(observations)
        Qprimes = self._model.predict(observations_next)
        for i in range(self._learning_batch_size):
            if dones[i]:
                Qvalues[i, actions[i]] = rewards[i]
            else:
                Qvalues[i, actions[i]] = rewards[i] + self._gamma * np.amax(Qprimes[i])
        self._model.fit(observations, Qvalues, epochs=1, verbose=0)

    def _decay_epsilon(self):
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)