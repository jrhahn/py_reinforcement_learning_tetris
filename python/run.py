import pandas as pd
import arcade
import tetris_block
from tetris_manager import TetrisManager
import tensorflow as tf

from ai import AI

SCREEN_WIDTH = 320
SCREEN_HEIGHT = 480

import time


class NoDisplayTetris(object):
    arcade_key_left = 'akleft'
    arcade_key_right = 'akright'
    arcade_key_up = 'akup'

    l_actions = [arcade_key_left, arcade_key_right, arcade_key_up]

    def __init__(self):

        self.do_use_player_control = False

        self.thresh_time_passed_input = 0
        self.thresh_time_passed_drop = 0

        self.tm = TetrisManager(None, SCREEN_WIDTH, SCREEN_HEIGHT,
                                thresh_time_passed_input=0,
                                thresh_time_passed_drop=0.001)

        self.ai = AI((self.tm.size_grid_x, self.tm.size_grid_y, 2), ArcadeTetris.l_actions, 0.0001)

    def update(self, delta_time):
        game_state = self.tm.update(delta_time)

        if not self.do_use_player_control:
            key = self.ai.get_action(self.tm)

            if key == NoDisplayTetris.arcade_key_left:
                self.tm.set_move(-1, 0)
            elif key == NoDisplayTetris.arcade_key_right:
                self.tm.set_move(1, 0)
            elif key == NoDisplayTetris.arcade_key_up:
                self.tm.rotate()

            self.ai.set_reward(self.tm, game_state, self.tm.score)

        if game_state == self.tm.GAME_STATE_END:
            self.ai.train()
            self.tm = TetrisManager(None, SCREEN_WIDTH, SCREEN_HEIGHT,
                                    thresh_time_passed_input=self.thresh_time_passed_input,
                                    thresh_time_passed_drop=self.thresh_time_passed_drop)


class ArcadeTetris(arcade.Window):
    l_actions = [arcade.key.LEFT, arcade.key.RIGHT, arcade.key.UP]

    def __init__(self):

        self.do_draw = True

        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT)
        # arcade.open_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Tetris")
        arcade.set_background_color(arcade.color.WHITE)

        self.set_update_rate(1 / 1000.0)

        self.image = arcade.Sprite("../resources/block.png", .25)

        self.do_use_player_control = False

        self.thresh_time_passed_input = 0
        self.thresh_time_passed_drop = 0

        if self.do_use_player_control:
            self.thresh_time_passed_input = 0.03
            self.thresh_time_passed_drop = 0.3

        self.tm = TetrisManager(self.image, SCREEN_WIDTH, SCREEN_HEIGHT,
                                thresh_time_passed_input=self.thresh_time_passed_input,
                                thresh_time_passed_drop=self.thresh_time_passed_drop)

        self.ai = AI((self.tm.size_grid_x, self.tm.size_grid_y, 2), ArcadeTetris.l_actions, 0.0001)

    def on_draw(self):
        if self.do_draw:
            arcade.start_render()
            self.tm.draw()
            arcade.finish_render()

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """

        if self.do_use_player_control:
            if key == arcade.key.LEFT:
                self.tm.set_move(-1, 0)
            elif key == arcade.key.RIGHT:
                self.tm.set_move(1, 0)
            elif key == arcade.key.UP:
                self.tm.rotate()

    def update(self, delta_time):
        game_state = self.tm.update(delta_time)

        if not self.do_use_player_control:
            key = self.ai.get_action(self.tm)

            if key == arcade.key.LEFT:
                self.tm.set_move(-1, 0)
            elif key == arcade.key.RIGHT:
                self.tm.set_move(1, 0)
            elif key == arcade.key.UP:
                self.tm.rotate()

            self.ai.set_reward(self.tm, game_state, self.tm.score)

        if game_state == self.tm.GAME_STATE_END:
            self.ai.train()
            self.tm = TetrisManager(self.image, SCREEN_WIDTH, SCREEN_HEIGHT,
                                    thresh_time_passed_input=self.thresh_time_passed_input,
                                    thresh_time_passed_drop=self.thresh_time_passed_drop)


def run():
    # Reset the graph
    tf.reset_default_graph()

    # game = ArcadeTetris()
    # arcade.run()

    game = NoDisplayTetris()

    duration = 0
    while True:
        start_time = time.time()
        game.update(delta_time=duration)
        duration = time.time() - start_time

        # print("{}".format(1/float(duration)))


if __name__ == "__main__":
    run()
