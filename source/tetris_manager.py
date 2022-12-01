import numpy as np

from tetris_block import TetrisBlock

import logging

from data_types.game_state import GameState

logger = logging.getLogger(__name__)


class TetrisManager:
    def __init__(self, image, screen_width: int, screen_height: int, **kwargs):
        self.image = image

        if image is not None:
            self.size_grid_x = int(screen_width / image.width)
            self.size_grid_y = int(screen_height / image.height)

            self.screen_height = self.size_grid_y * image.height
        else:
            image_height = kwargs.pop("image_height", 21.25)

            self.size_grid_x = int(screen_width / kwargs.pop("image_width", 21.25))
            self.size_grid_y = int(screen_height / image_height)

            self.screen_height = self.size_grid_y * image_height

        self.n_array = np.zeros((self.size_grid_x, self.size_grid_y))

        self.block_active = None
        self.score = 0

        self.tetris_block_parameters = kwargs

        self.create_new_block()

    def get_state(self):
        return self.block_active.update_grid(self.n_array, do_copy=True)

    def create_new_block(self):
        self.block_active = TetrisBlock(
            self.image,
            self.size_grid_x,
            self.size_grid_y,
            self.screen_height,
            **self.tetris_block_parameters
        )

    def draw(self):
        self.block_active.draw()

        for x in range(self.size_grid_x):
            for y in range(self.size_grid_y):
                if self.n_array[x, y] == 1:
                    self.image.center_x = (x + 0.5) * self.image.width
                    self.image.center_y = (y + 0.5) * self.image.height
                    self.image.draw()
                    # canvas.drawBitmap(self.image, x * self.image.getWidth(), y * self.image.getHeight(), None)

    def set_move(self, move_x, move_y):
        self.block_active.set_moving_vector(move_x, move_y)

    def rotate(self):
        self.block_active.rotate()

    def get_block_size(self):
        return self.image.getWidth()

    def get_active_pos_x(self):
        return self.block_active.get_x()

    def get_active_pos_y(self):
        return self.block_active.get_y()

    def check_if_line_is_full(self):
        # for y in range(self.size_grid_y - 1, -1, -1):
        for y in range(self.size_grid_y):
            is_complete = (self.n_array[:, y] > 0).all()

            self.score += (
                    self.n_array[:, y].sum()
                    * (self.size_grid_y - y)
                    / float(self.size_grid_y)
            )

            if is_complete:
                self.score += 1000
                logger.info(f"LINE IS FULL Score: {self.score}")
                # move lines
                self.n_array[:, y:-1] = self.n_array[:, y + 1:]

    def update(self, delta_time):
        self.n_array = self.block_active.update(self.n_array, delta_time)

        if not self.block_active.is_active:
            self.create_new_block()

            self.check_if_line_is_full()

        if self.n_array[:, -2].sum() > 0:
            self.score -= 500
            logger.info(f"Score: {self.score}")
            logger.info("***NEW GAME***")

            return GameState.End

        return GameState.Ok
