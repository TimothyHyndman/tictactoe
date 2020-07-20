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
