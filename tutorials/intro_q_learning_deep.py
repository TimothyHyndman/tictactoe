"""
https://rubikscode.net/2019/07/08/deep-q-learning-with-python-and-tensorflow-2-0/
https://theaisummer.com/Deep_Q_Learning/
"""

import numpy as np
import random
from collections import deque
import gym

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

MODEL_LOCATION = "checkpoint"


class Agent:
    def __init__(self, env, optimizer, pre_trained_model=None):
        self._state_size = env.observation_space.n
        self._action_size = env.action_space.n
        self._optimizer = optimizer

        # A deque (double ended queue) is similar to a list but provides faster pop and
        # append methods since it can preallocate the memory. When the deque is full,
        # adding a new item to one end removes an item from the other end.
        self.experience_replay = deque(maxlen=2000)

        self.gamma = 0.6
        self.epsilon = 0.1

        if pre_trained_model is not None:
            self.q_network = load_model(pre_trained_model)
            self.target_network = load_model(pre_trained_model)
        else:
            self.q_network = self._build_compile_model()
            self.target_network = self._build_compile_model()
            self.align_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))

    def _build_compile_model(self):
        model = Sequential([
            Embedding(input_dim=self._state_size, output_dim=10, input_length=1),
            Reshape((10,)),
            Dense(50, activation='relu'),
            Dense(50, activation='relu'),
            Dense(self._action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def align_target_model(self):
        """Copies the parameters from q_network to target_network"""
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state, env):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()

        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)

        for state, action, reward, next_state, terminated in minibatch:
            target = self.q_network.predict(state)
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.max(t)

            # TODO: Why are we training one observation at a time?
            self.q_network.fit(state, target, epochs=1, verbose=0)


def main():
    env = gym.make("Taxi-v3").env
    env.render()

    optimizer = Adam(learning_rate=0.01)
    agent = Agent(env, optimizer, pre_trained_model=MODEL_LOCATION)

    agent.q_network.summary()

    batch_size = 32
    num_of_episodes = 1000
    timesteps_per_episode = 1000

    for episode in range(0, num_of_episodes):
        print(episode)
        state = env.reset()
        state = np.reshape(state, (1, 1))

        for timestep in range(timesteps_per_episode):
            action = agent.act(state, env)
            next_state, reward, terminated, info = env.step(action)
            next_state = np.reshape(next_state, (1, 1))
            agent.store(state, action, reward, next_state, terminated)

            state = next_state

            if terminated:
                print("I actually did something useful for once")
                agent.align_target_model()
                break

        if len(agent.experience_replay) > batch_size:
            agent.retrain(batch_size)

        if (episode + 1) % 10 == 0:
            print("**********************************")
            print("Episode: {}".format(episode + 1))
            env.render()
            agent.q_network.save(MODEL_LOCATION)
            print("**********************************")


if __name__ == '__main__':
    main()
