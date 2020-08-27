"""
Core code for tic-tac-toe
"""
import numpy as np


class Game:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.winner = None
        self.draw = None
        self.xo = 1

    def play_move(self, row: int, col: int):
        if self.board[row, col] == 0:
            self.board[row, col] = self.xo
            self.xo = -self.xo

    def check_win(self):
        if (self.board.sum(0) == 3).any():
            self.winner = 1
        if (self.board.sum(1) == 3).any():
            self.winner = 1
        if (self.board.sum(0) == -3).any():
            self.winner = -1
        if (self.board.sum(1) == -3).any():
            self.winner = -1

        if self.board.diagonal().sum() == 3:
            self.winner = 1
        if self.board.diagonal().sum() == -3:
            self.winner = -1
        if np.fliplr(self.board).diagonal().sum() == 3:
            self.winner = 1
        if np.fliplr(self.board).diagonal().sum() == -3:
            self.winner = -1

        if (self.board != 0).all() and self.winner is None:
            self.draw = True


WIN_REWARD = 10
DRAW_REWARD = 0


class GameEnv(Game):
    """A wrapper around Game inspired by the OpenAI gym environments"""
    def state(self):
        """
        Returns stack of 3x3 matrices.
        First is player 1's pieces, then player 2's pieces.
        """
        stack = np.dstack([self.board == 1, self.board == -1])
        return stack

    def step(self, action):
        row = action // 3
        col = action - 3 * row
        self.play_move(row, col)
        self.check_win()

        reward = 0
        if self.winner:
            reward = WIN_REWARD
        if self.draw:
            reward = DRAW_REWARD

        return self.state(), reward, self.draw or self.winner

    def possible_actions(self):
        return self.board == 0

    def render(self):
        print(f'{self.board[0][0]}|{self.board[0][1]}|{self.board[0][2]}')
        print(f'{self.board[1][0]}|{self.board[1][1]}|{self.board[1][2]}')
        print(f'{self.board[2][0]}|{self.board[2][1]}|{self.board[2][2]}')
