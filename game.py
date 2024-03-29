"""
Core code for tic-tac-toe
"""
import numpy as np

WIN_REWARD = 10
DRAW_REWARD = 0


class Game:
    def __init__(self, player1=None, player2=None):
        self.board = np.zeros((3, 3), dtype=int)
        self.winner = None
        self.draw = None
        self.xo = 1
        self.player1 = player1
        self.player2 = player2

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


class GameEnv(Game):
    """A wrapper around Game inspired by the OpenAI gym environments"""
    def state(self, player=None):
        """
        Returns flattened stack of 3x3 matrices.
        First is player's pieces, then opponent's pieces,
        Finally a single element: 1 for player 1 or 0 for player 2
        """
        if not player:
            player = self.xo

        state = np.hstack([(self.board == player).flatten(), (self.board == -player).flatten(), player == 1])
        return state

    def possible_actions(self):
        return (self.board == 0).flatten()

    def render(self):
        print(f'{self.board[0][0]}|{self.board[0][1]}|{self.board[0][2]}')
        print(f'{self.board[1][0]}|{self.board[1][1]}|{self.board[1][2]}')
        print(f'{self.board[2][0]}|{self.board[2][1]}|{self.board[2][2]}')
