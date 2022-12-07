#!/usr/bin/env python3
import logging
import time

import arcade

from tetris.arcade import ArcadeTetris
from tetris.no_display import NoDisplayTetris

logging.basicConfig(level=logging.INFO)


def run():
    # Reset the graph
    # tf.reset_default_graph()

    # game = ArcadeTetris(screen_width=320, screen_height=480, )
    # arcade.run()

    game = NoDisplayTetris(screen_width=320, screen_height=480, )

    duration = 0
    while True:
        start_time = time.time()
        game.update(delta_time=duration)
        duration = time.time() - start_time

        # print("{}".format(1/float(duration)))


if __name__ == "__main__":
    run()
