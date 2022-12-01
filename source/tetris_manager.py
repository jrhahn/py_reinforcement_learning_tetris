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

        self._array = np.zeros((self.size_grid_x, self.size_grid_y))

        self.block_active = None
        self.score = 0

        self.tetris_block_parameters = kwargs

        self.create_new_block()

    @property
    def state(self) -> np.array:
        return self.block_active.update_grid(self._array, do_copy=True,)

    def create_new_block(self):
        self.block_active = TetrisBlock(
            image=self.image,
            size_grid_x=self.size_grid_x,
            size_grid_y=self.size_grid_y,
            screen_height=self.screen_height,
            **self.tetris_block_parameters
        )

    def draw(self):
        self.block_active.draw()

        for x in range(self.size_grid_x):
            for y in range(self.size_grid_y):
                if self._array[x, y] == 1:
                    self.image.center_x = (x + 0.5) * self.image.width
                    self.image.center_y = (y + 0.5) * self.image.height
                    self.image.draw()
                    # canvas.drawBitmap(self.image, x * self.image.getWidth(), y * self.image.getHeight(), None)

    def set_move(self, move_x, move_y):
        self.block_active.set_moving_vector(move_x, move_y)

    def rotate(self):
        self.block_active.rotate()

    @property
    def block_size(self) -> int:
        return self.image.getWidth()

    @property
    def active_block_pos_x(self) -> int:
        return self.block_active.x

    @property
    def active_block_pos_y(self) -> int:
        return self.block_active.y

    def check_if_line_is_full(self) -> None:
        # for y in range(self.size_grid_y - 1, -1, -1):
        for y in range(self.size_grid_y):
            is_complete = (self._array[:, y] > 0).all()

            self.score += (
                    self._array[:, y].sum()
                    * (self.size_grid_y - y)
                    / float(self.size_grid_y)
            )

            if is_complete:
                self.score += 1000
                logger.info(f"LINE IS FULL Score: {self.score}")
                # move lines
                self._array[:, y:-1] = self._array[:, y + 1:]

    def update(self, delta_time: float) -> GameState:
        self._array = self.block_active.update(self._array, delta_time)

        if not self.block_active.is_active:
            self.create_new_block()

            self.check_if_line_is_full()

        if self._array[:, -2].sum() > 0:
            self.score -= 500
            logger.info(f"Score: {self.score}")
            logger.info("***NEW GAME***")

            return GameState.End

        return GameState.Ok
