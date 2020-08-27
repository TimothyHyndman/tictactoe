"""
Deep Q learning AI for tic-tac-toe
"""
from collections import deque
import random
import tensorflow as tf

from game import GameEnv
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import load_model


# Player 1 plays
# Player 2 plays and gives reward to player 1
# Player 1 plays and gives reward to player 2
# .
# .
# .
# Player x plays and game terminated. Both players get reward.

# Simple approach
# Player 1 plays followed immediately by random move, then give reward to player 1.


class TinyBrain:
    """
    Randomly chooses tic-tac-toe moves and doesn't learn.
    """
    def __init__(self):
        self._board_size = (3, 3)

    def act(self, possible_actions):
        values = np.random.random(self._board_size)
        # Don't move where not possible to play
        values = values * possible_actions
        row, col = np.unravel_index(np.argmax(values), values.shape)
        action = (row, col)
        return action


class BigBrain:
    """
    Defines an agent to learn and play tic-tac-toe
    """
    def __init__(self):
        self._action_size = 9  # there are 9 possible moves in tic-tac-toe
        self._input_shape = (3, 3, 2)  # 2 lots of a 3x3 board (one for each player's moves)
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_target_model()
        self.experience_replay = deque(maxlen=2000)
        self.gamma = 0.6

    def _build_compile_model(self):
        """
        Builds and compiles a fully connected NN with two hidden layers and a softmax
        output to represent a probability distribution on moves to play.
        """
        model = Sequential([
            Input(shape=self._input_shape),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(self._action_size)
        ])
        model.compile(loss='mse', optimizer="adam")
        return model

    def align_target_model(self):
        """Copies the parameters from q_network to target_network"""
        self.target_network.set_weights(self.q_network.get_weights())

    def move_probabilities(self, state, possible_actions):
        # Need to add "batch" dimension
        state_reshape = np.expand_dims(state, axis=0)
        predictions = self.q_network.predict(state_reshape)

        probabilities = tf.nn.softmax(predictions).numpy()
        probabilities = probabilities * possible_actions.flatten()
        probabilities = probabilities / np.sum(probabilities)

        return np.reshape(probabilities, (3, 3))

    def act(self, state, possible_actions):
        probabilities = self.move_probabilities(state, possible_actions)
        row, col = np.unravel_index(np.argmax(probabilities), probabilities.shape)
        action = (row, col)
        return action

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, min(batch_size, len(self.experience_replay)))

        for state, action, reward, next_state, terminated in minibatch:
            state_reshape = np.expand_dims(state, axis=0)
            target = self.q_network.predict(state_reshape)
            target_reshape = np.reshape(target, (3, 3))
            if terminated:
                target_reshape[action] = reward
            else:
                next_state_reshape = np.expand_dims(state, axis=0)
                t = self.target_network.predict(next_state_reshape)
                target_reshape[action] = reward + self.gamma * np.max(t)

            target = np.reshape(target_reshape, (1, 9))
            # TODO: Why are we training one observation at a time?
            self.q_network.fit(state_reshape, target, epochs=1, verbose=0)


WIN_REWARD = 10
DRAW_REWARD = -1
LOSS_REWARD = -10


def main():
    big_brain = BigBrain()
    tiny_brain = TinyBrain()

    episode = 1
    while True:
        env = GameEnv()
        while True:
            state = env.state()
            possible_actions = env.possible_actions()
            move1 = big_brain.act(state, possible_actions)
            x, y = move1
            env.play_move(x, y)
            env.check_win()

            if env.winner:
                big_brain.store(state, move1, reward=WIN_REWARD, next_state=env.state(), terminated=True)
                break
            if env.draw:
                big_brain.store(state, move1, reward=DRAW_REWARD, next_state=env.state(), terminated=True)
                break

            possible_actions = env.possible_actions()
            x, y = tiny_brain.act(possible_actions)
            env.play_move(x, y)
            env.check_win()

            if env.winner:
                big_brain.store(state, move1, reward=LOSS_REWARD, next_state=env.state(), terminated=True)
                break
            if env.draw:
                big_brain.store(state, move1, reward=DRAW_REWARD, next_state=env.state(), terminated=True)
                break

            # No result
            big_brain.store(state, move1, reward=0, next_state=env.state(), terminated=False)

        print(env.winner)
        big_brain.retrain(batch_size=32)
        if episode % 10 == 0:
            print("aligning target model")
            big_brain.align_target_model()
        episode += 1


if __name__ == '__main__':
    main()
