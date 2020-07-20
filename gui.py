"""
GUI for tic-tac-toe

Adapted from https://www.geeksforgeeks.org/tic-tac-toe-gui-in-python-using-pygame/
"""

import sys
import math
import pygame as pg
from typing import Tuple
from game import Game


# declaring the global variables
width = 400
height = 400
white = (255, 255, 255)
x_positions = {0: 30, 1: int(width / 3) + 30, 2: int(width / 3) * 2 + 30}
y_positions = {0: 30, 1: int(height / 3) + 30, 2: int(height / 3) * 2 + 30}
line_color = (0, 0, 0)
fps = 144

# initializing the pygame window
pg.init()
CLOCK = pg.time.Clock()
screen = pg.display.set_mode((width, height + 100), 0, 32)
pg.display.set_caption("Tic Tac Toe")

# loading the images as python object
x_img = pg.image.load("images/X_modified.png")
y_img = pg.image.load("images/o_modified.png")
x_img = pg.transform.scale(x_img, (80, 80))
o_img = pg.transform.scale(y_img, (80, 80))


def draw_status(game):

    if game.winner is None:
        message = f"Player {game.xo}'s Turn"
    else:
        message = f"Player {game.winner} won!"
    if game.draw:
        message = "Game Drawn"

    # setting a font object
    font = pg.font.Font(None, 30)

    # setting the font properties like
    # color and width of the text
    text = font.render(message, 1, (255, 255, 255))

    # copy the rendered message onto the board
    # creating a small block at the bottom of the main display
    screen.fill((0, 0, 0), (0, 400, 500, 100))
    text_rect = text.get_rect(center=(int(width / 2), 500 - 50))
    screen.blit(text, text_rect)
    pg.display.update()


def draw_xo(row, col, xo):
    if row in y_positions.keys() and col in x_positions.keys():
        posy = y_positions[row]
        posx = x_positions[col]
        if xo == 1:
            screen.blit(x_img, (posy, posx))
        elif xo == -1:
            screen.blit(o_img, (posy, posx))


def draw_board(board):
    screen.fill(white)

    # drawing vertical lines
    pg.draw.line(screen, line_color, (int(width / 3), 0), (int(width / 3), height), 7)
    pg.draw.line(screen, line_color, (int(width / 3 * 2), 0), (int(width / 3 * 2), height), 7)

    # drawing horizontal lines
    pg.draw.line(screen, line_color, (0, int(height / 3)), (width, int(height / 3)), 7)
    pg.draw.line(screen, line_color, (0, int(height / 3 * 2)), (width, int(height / 3 * 2)), 7)

    for row in range(3):
        for column in range(3):
            draw_xo(row, column, board[row, column])


def user_click() -> Tuple[int, int]:
    x, y = pg.mouse.get_pos()
    col = math.floor(3 * x / width)
    row = math.floor(3 * y / width)

    return col, row


if __name__ == '__main__':
    current_game = Game()
    draw_board(current_game.board)
    draw_status(current_game)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif event.type is pg.MOUSEBUTTONDOWN:
                x, y = user_click()
                current_game.play_move(x, y)
                current_game.check_win()
                draw_board(current_game.board)
                draw_status(current_game)
                if current_game.winner or current_game.draw:
                    current_game = Game()
        pg.display.update()
        CLOCK.tick(fps)
