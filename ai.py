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


WIN_REWARD = 1
DRAW_REWARD = 0
LOSS_REWARD = -5


class TinyBrain:
    """
    Randomly chooses tic-tac-toe moves and doesn't learn.
    """
    def __init__(self):
        self._board_size = 9

    def select_move(self, game):
        possible_actions = game.possible_actions()
        values = np.random.random(self._board_size)
        # Don't move where not possible to play
        values = values * possible_actions
        row, col = np.unravel_index(np.argmax(values), (3, 3))
        action = (row, col)
        return action


class BigBrain:
    """
    Defines an agent to learn and play tic-tac-toe
    """
    def __init__(self, load_model=None, tryhard_mode=True):
        self._action_size = 9  # there are 9 possible moves in tic-tac-toe
        # 2 lots of a 3x3 board (one for each player's moves) plus constant valued plane
        # representing whose turn it is
        self._input_shape = (19,)
        self.experience_replay = deque(maxlen=100*5)  # past results last no more than 100 games
        self.gamma = 0.6
        self.tryhard_mode = tryhard_mode
        self.alpha = 0.2

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
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            # Dense(32, activation='relu'),
            # Dense(32, activation='relu'),
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
        predictions = self.q_network(state_reshape, training=False)

        probabilities = tf.nn.softmax(predictions).numpy()
        probabilities = probabilities * possible_actions
        probabilities = probabilities / np.sum(probabilities)

        return np.reshape(probabilities, (3, 3))

    def select_move(self, game, explore=None):
        state = game.state()
        possible_actions = game.possible_actions()
        probabilities = self.move_probabilities(state, possible_actions)

        if explore is None:
            explore = not self.tryhard_mode
        if not explore:
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
            target = self.q_network(state_reshape, training=False).numpy()
            target_reshape = np.reshape(target, (3, 3))
            if terminated:
                target_reshape[action] = reward
            else:
                next_state_reshape = np.expand_dims(next_state, axis=0)
                next_state_action_values = self.target_network(next_state_reshape, training=False).numpy().flatten()
                possible_next_state_actions_values = next_state_action_values[possible_actions]
                target_reshape[action] = self.gamma * np.max(possible_next_state_actions_values)

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


def print_diagnostics(candidate_player, results, episode):
    test_env_start = GameEnv()
    test_env_about_to_win = GameEnv()
    test_env_about_to_win.play_move(0, 0)
    test_env_about_to_win.play_move(2, 2)
    test_env_about_to_win.play_move(0, 1)
    test_env_about_to_win.play_move(1, 1)

    test_env_about_to_win_p2 = GameEnv()
    test_env_about_to_win_p2.play_move(0, 0)
    test_env_about_to_win_p2.play_move(2, 2)
    test_env_about_to_win_p2.play_move(1, 1)
    test_env_about_to_win_p2.play_move(2, 1)
    test_env_about_to_win_p2.play_move(1, 0)

    no_results = 100
    denom = len(results[-no_results:])
    wins = len([res for res in results[-no_results:] if res == 1])
    draws = len([res for res in results[-no_results:] if res == 0])
    losses = len([res for res in results[-no_results:] if res == -1])
    print(f"Win rate: {wins / denom}")
    print(f"Loss rate: {losses / denom}")
    print(f"Draw rate: {draws / denom}")
    initial_preferences = candidate_player.move_probabilities(test_env_start.state(), test_env_start.possible_actions())
    about_to_win_preferences = candidate_player.move_probabilities(test_env_about_to_win.state(),
                                                                   test_env_about_to_win.possible_actions())
    about_to_win_preferences_p2 = candidate_player.move_probabilities(
        test_env_about_to_win_p2.state(),
        test_env_about_to_win_p2.possible_actions()
    )
    print(f"Initial move preference after {episode} games")
    print(initial_preferences)
    print(f"Preferences when winning in top right")
    print(about_to_win_preferences)
    print(f"Preferences when winning in bottom left")
    print(about_to_win_preferences_p2)


def evaluate_candidate(candidate_player, reference_player):
    wins = 0
    draws = 0
    losses = 0

    for _ in range(400):
        env = GameEnv()
        current_player = candidate_player if random.random() < 0.5 else reference_player
        while True:
            move = current_player.select_move(env)
            env.play_move(*move)  # Current player flips
            env.check_win()
            if env.winner and current_player is candidate_player:
                wins += 1
                break
            elif env.winner and current_player is reference_player:
                losses += 1
                break
            elif env.draw:
                draws += 1
                break
            else:
                current_player = reference_player if current_player is candidate_player else candidate_player

    return wins, draws, losses


def main():
    # reference_player = BigBrain(tryhard_mode=False)
    model_name = "models/model_010_64_64_64_self_play.h5"
    model_name_reference = "models/reference.h5"
    reference_player = TinyBrain()
    # reference_player = BigBrain(load_model=model_name, tryhard_mode=False)  # For continuing training
    # candidate_player = BigBrain(load_model=model_name, tryhard_mode=False)  # For continuing training
    candidate_player = BigBrain(tryhard_mode=False)  # For starting training

    episode = 1
    results = []

    while True:
        env = GameEnv()
        first_move = True
        # immediate store is all the information we have immediately after a move
        # (current state, possible actions, move)
        immediate_store = []
        # delayed store is all the information we get after the next move
        # (next state, reward, terminated)
        delayed_store = []

        # Randomly choose who goes first
        current_player = candidate_player if random.random() < 0.5 else reference_player
        # current_player = candidate_player  # just start by training for playing first

        while True:
            state = env.state()
            move = current_player.select_move(env)
            if current_player is candidate_player:
                if first_move:
                    first_move = False
                else:
                    # Finish providing information for candidate player's last move
                    possible_actions = env.possible_actions()
                    delayed_store.append((possible_actions, 0, state, False))
                # Provide starting information for candidate player's current move
                do_explore = random.random() < 0.3
                move = current_player.select_move(env, explore=do_explore)
                immediate_store.append((state, move, do_explore))

            env.play_move(*move)  # Current player flips
            env.check_win()

            if env.winner or env.draw:
                # If game has ended we need to give rewards to both players
                if env.draw:
                    delayed_store.append((None, DRAW_REWARD, None, True))
                    results.append(0)
                elif current_player is candidate_player:
                    # Winner is always whoever played last
                    delayed_store.append((None, WIN_REWARD, None, True))
                    results.append(1)
                else:
                    delayed_store.append((None, LOSS_REWARD, None, True))
                    results.append(-1)

                for immediate, delayed in zip(immediate_store, delayed_store):
                    state, move, do_explore = immediate
                    if not do_explore:
                        candidate_player.store(state, move, *delayed)
                break

            current_player = reference_player if current_player is candidate_player else candidate_player

        if episode % 100 == 0:
            print(episode)
        if episode % 100 == 0:
            print("Training")
            candidate_player.retrain(batch_size=50)
            print("Aligning target model")
            candidate_player.align_target_model()
            print("Saving model")
            candidate_player.save(model_name)
            print_diagnostics(candidate_player, results, episode)

        # if episode % 1000 == 0:
        #     print("Evaluating candidate against reference")
        #     # print_diagnostics(candidate_player, results, episode)
        #      # See if we can replace reference model with our new one
        #     # candidate_player.tryhard_mode = True
        #     wins, draws, losses = evaluate_candidate(candidate_player, reference_player)
        #     print(f"{wins}, {draws}, {losses}")
        #     if wins + losses > 0:
        #         percentage_wins = wins / (wins + losses)
        #     else:
        #         percentage_wins = 0
        #     print(f"percentage wins: {percentage_wins}")
        #
        #     if percentage_wins > 0.55:
        #         print("Candidate promoted to reference")
        #         reference_player = candidate_player
        #         candidate_player.save(model_name_reference)
        #         candidate_player = BigBrain(tryhard_mode=False)
        #         candidate_player.q_network = reference_player.q_network
        #         candidate_player.target_network = reference_player.target_network

        episode += 1


if __name__ == '__main__':
    main()
