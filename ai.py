"""
Deep Q learning AI for tic-tac-toe
"""
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


def main():
    env = GameEnv()
    big_brain = BigBrain()
    tiny_brain = TinyBrain()

    state = env.state()
    possible_actions = env.possible_actions()

    big_brain.act(state, possible_actions)
    tiny_brain.act(possible_actions)

    env.render()
    time.sleep(1)

    while True:
        state = env.state()
        possible_actions = env.possible_actions()
        x, y = big_brain.act(state, possible_actions)
        env.play_move(x, y)
        env.check_win()
        env.render()
        time.sleep(1)

        if env.winner or env.draw:
            break

        possible_actions = env.possible_actions()
        x, y = tiny_brain.act(possible_actions)
        env.play_move(x, y)
        env.check_win()
        env.render()
        time.sleep(1)

        if env.winner or env.draw:
            break


if __name__ == '__main__':
    main()
