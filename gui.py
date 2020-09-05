"""
GUI for tic-tac-toe

Adapted from https://www.geeksforgeeks.org/tic-tac-toe-gui-in-python-using-pygame/
"""

import sys
import pygame as pg

from game import GameEnv
from players import HumanPlayer, AIPlayer

white = (255, 255, 255)


class GameGUI(GameEnv):
    def __init__(self, player1, player2):
        super().__init__()
        self.width = 400
        self.height = 400
        self.x_positions = {0: 30, 1: int(self.width / 3) + 30, 2: int(self.width / 3) * 2 + 30}
        self.y_positions = {0: 30, 1: int(self.height / 3) + 30, 2: int(self.height / 3) * 2 + 30}
        self.line_color = (0, 0, 0)
        self.fps = 144

        self.player1 = player1
        self.player2 = player2

        # initializing the pygame window
        pg.init()
        self.CLOCK = pg.time.Clock()
        self.screen = pg.display.set_mode((self.width, self.height + 100), 0, 32)
        pg.display.set_caption("Tic Tac Toe")

        # loading the images as python object
        x_img = pg.image.load("images/X_modified.png")
        y_img = pg.image.load("images/o_modified.png")
        self.x_img = pg.transform.scale(x_img, (80, 80))
        self.o_img = pg.transform.scale(y_img, (80, 80))

    def draw_status(self):

        names = {1: self.player1.name, -1: self.player2.name}

        if self.winner is None:
            message = f"{names[self.xo]}'s Turn"
        else:
            message = f"{names[self.winner]} won!"
        if self.draw:
            message = "Game Drawn"

        # setting a font object
        font = pg.font.Font(None, 30)

        # setting the font properties like
        # color and width of the text
        text = font.render(message, 1, (255, 255, 255))

        # copy the rendered message onto the board
        # creating a small block at the bottom of the main display
        self.screen.fill((0, 0, 0), (0, 400, 500, 100))
        text_rect = text.get_rect(center=(int(self.width / 2), 500 - 50))
        self.screen.blit(text, text_rect)
        pg.display.update()

    def draw_xo(self, row, col, xo):
        if row in self.y_positions.keys() and col in self.x_positions.keys():
            posy = self.y_positions[row]
            posx = self.x_positions[col]
            if xo == 1:
                self.screen.blit(self.x_img, (posy, posx))
            elif xo == -1:
                self.screen.blit(self.o_img, (posy, posx))

    def draw_board(self):
        self.screen.fill(white)

        # drawing vertical lines
        pg.draw.line(
            self.screen,
            self.line_color,
            (int(self.width / 3), 0),
            (int(self.width / 3), self.height),
            7
        )
        pg.draw.line(
            self.screen,
            self.line_color,
            (int(self.width / 3 * 2), 0),
            (int(self.width / 3 * 2), self.height),
            7
        )

        # drawing horizontal lines
        pg.draw.line(
            self.screen,
            self.line_color,
            (0, int(self.height / 3)),
            (self.width, int(self.height / 3)),
            7
        )
        pg.draw.line(
            self.screen,
            self.line_color,
            (0, int(self.height / 3 * 2)),
            (self.width, int(self.height / 3 * 2)),
            7
        )

        for row in range(3):
            for column in range(3):
                self.draw_xo(row, column, self.board[row, column])

    def game_loop(self):
        self.draw_board()
        self.draw_status()
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
            if not (self.winner or self.draw):
                if self.xo == 1:
                    x, y = self.player1.select_move(self)
                    self.play_move(x, y)
                elif self.xo == -1:
                    x, y = self.player2.select_move(self)
                    self.play_move(x, y)
                self.check_win()
                self.draw_board()
                self.draw_status()
                pg.display.update()
            self.CLOCK.tick(self.fps)  # TODO: Figure out how this fits in. Do I need it?


if __name__ == '__main__':
    # p1 = HumanPlayer(name='Tim')
    p2 = AIPlayer(name="Fred")
    p1 = HumanPlayer(name='Alex')
    current_game = GameGUI(player1=p1, player2=p2)
    current_game.game_loop()
