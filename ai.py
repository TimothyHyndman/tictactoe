"""
Deep Q learning AI for tic-tac-toe
"""

from collections import deque
import random

from game import GameEnv
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten


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

    def select_move(self, game):
        possible_actions = game.possible_actions()
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
    def __init__(self, tryhard_mode=False, load_model=None):
        self._action_size = 9  # there are 9 possible moves in tic-tac-toe
        self._input_shape = (3, 3, 2)  # 2 lots of a 3x3 board (one for each player's moves)
        self.experience_replay = deque(maxlen=5 * 40)  # past results last no more than 40 games
        self.gamma = 0.6
        self.tryhard_mode = tryhard_mode

        if load_model:
            self.load(load_model)
        else:
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

    def select_move(self, game):
        state = game.state()
        possible_actions = game.possible_actions()
        probabilities = self.move_probabilities(state, possible_actions)

        if self.tryhard_mode:
            # Always select highest probability move.
            row, col = np.unravel_index(np.argmax(probabilities), probabilities.shape)
        else:
            # Random select move with probabilities as given
            row, col = np.unravel_index(
                np.argmax(~(np.cumsum(probabilities.flatten()) < np.random.rand())),
                probabilities.shape
            )

        action = (row, col)
        return action

    def store(self, state, action, possible_actions, reward, next_state, terminated):
        self.experience_replay.append((state, action, possible_actions, reward, next_state, terminated))

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)

        states = []
        targets = []
        for state, action, possible_actions, reward, next_state, terminated in minibatch:
            state_reshape = np.expand_dims(state, axis=0)
            target = self.q_network.predict(state_reshape)
            target_reshape = np.reshape(target, (3, 3))
            if terminated:
                target_reshape[action] = reward
            else:
                next_state_reshape = np.expand_dims(state, axis=0)

                t = self.target_network.predict(next_state_reshape)
                target_reshape[action] = reward + self.gamma * np.max(t.flatten()[possible_actions.flatten()])

            target = np.reshape(target_reshape, (1, 9))

            states.append(state_reshape)
            targets.append(target)

        states = np.vstack(states)
        targets = np.vstack(targets)
        self.q_network.fit(states, targets, verbose=0)

    def save(self, filename='my_model.h5'):
        self.q_network.save(filename)

    def load(self, filename='my_model.h5'):
        self.q_network = tf.keras.models.load_model(filename)


WIN_REWARD = 10
DRAW_REWARD = 0
LOSS_REWARD = -50


def main():
    initial_preferences = None

    tiny_brain = TinyBrain()
    model_name = "models/model_002_32_32_random_opponent.h5"
    # big_brain = BigBrain(tryhard_mode=True, load_model=model_name)
    big_brain = BigBrain()

    episode = 1
    results = []
    while True:
        env = GameEnv()
        if not big_brain.tryhard_mode:
            initial_preferences = big_brain.move_probabilities(env.state(), env.possible_actions())
        while True:
            # For storing
            state = env.state()
            possible_actions = env.possible_actions()

            move1 = big_brain.select_move(env)
            x, y = move1
            env.play_move(x, y)
            env.check_win()

            if env.winner:
                results.append(1)
                big_brain.store(state, move1, possible_actions, reward=WIN_REWARD, next_state=env.state(), terminated=True)
                break
            if env.draw:
                results.append(0)
                big_brain.store(state, move1, possible_actions, reward=DRAW_REWARD, next_state=env.state(), terminated=True)
                break

            x, y = tiny_brain.select_move(env)
            env.play_move(x, y)
            env.check_win()

            if env.winner:
                results.append(-1)
                big_brain.store(state, move1, possible_actions, reward=LOSS_REWARD, next_state=env.state(), terminated=True)
                break
            if env.draw:
                results.append(0)
                big_brain.store(state, move1, reward=DRAW_REWARD, next_state=env.state(), terminated=True)
                break

            # No result
            big_brain.store(state, move1, possible_actions, reward=0, next_state=env.state(), terminated=False)

        no_results = 100
        wins = len([res for res in results[-no_results:] if res == 1])
        draws = len([res for res in results[-no_results:] if res == 0])
        losses = len([res for res in results[-no_results:] if res == -1])
        print(f"Win rate: {wins / no_results}")
        print(f"Loss rate: {losses / no_results}")
        print(f"Draw rate: {draws / no_results}")

        if not big_brain.tryhard_mode:
            if len(big_brain.experience_replay) > 8:
                big_brain.retrain(batch_size=8)
            if episode % 10 == 0:
                print("Aligning target model")
                big_brain.align_target_model()
                print(f"Evaluation of starting move after {episode} games:")
                print(initial_preferences)
                print("Saving model")
                big_brain.save(model_name)

        episode += 1


if __name__ == '__main__':
    main()
