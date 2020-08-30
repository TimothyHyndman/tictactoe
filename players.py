"""
A class for a human player
"""
from typing import Tuple
import math
import pygame as pg
import sys

from ai import BigBrain


def user_click(width) -> Tuple[int, int]:
    x, y = pg.mouse.get_pos()
    col = math.floor(3 * x / width)
    row = math.floor(3 * y / width)

    return col, row


class Player:
    def __init__(self, name):
        self.name = name


class HumanPlayer(Player):
    @staticmethod
    def select_move(game):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                elif event.type is pg.MOUSEBUTTONDOWN:
                    x, y = user_click(game.width)
                    return x, y


class AIPlayer(Player):
    def __init__(self, name):
        super().__init__(name=name)
        self.ai = BigBrain(
            load_model="models/model_006_32_32_random_opponent_both_players.h5"
        )

    def select_move(self, game):
        return self.ai.select_move(game)
