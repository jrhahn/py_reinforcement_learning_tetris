import logging
from typing import Dict

import numpy as np

from data_types.block_type import BlockType

logger = logging.getLogger(__name__)


class TetrisBlock:
    L_array = [
        np.array([[1, 0], [1, 0], [1, 1]]),
        np.array([[1, 1, 1], [1, 0, 0]]),
        np.array([[1, 1], [0, 1], [0, 1]]),
        np.array(
            [
                [0, 0, 1],
                [1, 1, 1],
            ]
        ),
    ]
    L_inv_array = [
        np.array([[0, 1], [0, 1], [1, 1]]),
        np.array(
            [
                [1, 0, 0],
                [1, 1, 1],
            ]
        ),
        np.array([[1, 1], [1, 0], [1, 0]]),
        np.array([[1, 1, 1], [0, 0, 1]]),
    ]
    Triangle_array = [
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
            ]
        ),
        np.array([[1, 0], [1, 1], [1, 0]]),
        np.array([[1, 1, 1], [0, 1, 0]]),
        np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0]]),
    ]
    Block_array = [np.array([[1, 1], [1, 1]])]
    Bar_array = [
        np.array([[1, 1, 1, 1]]),
        np.array([[1], [1], [1], [1]]),
    ]
    Z_array = [np.array([[1, 1, 0], [0, 1, 1]]), np.array([[0, 1], [1, 1], [1, 0]])]
    S_array = [np.array([[0, 1, 1], [1, 1, 0]]), np.array([[1, 0], [1, 1], [0, 1]])]

    def __init__(
            self,
            image,
            size_grid_x: int,
            size_grid_y: int,
            screen_height: int,
            **kwargs: Dict,
    ) -> None:

        self.time_passed_input = 0
        self.time_passed_drop = 0
        self.image = image
        self.screen_height = screen_height

        self.size_grid_x = size_grid_x
        self.size_grid_y = size_grid_y

        # self.bb_left = 0
        # self.bb_right = 0
        # self.bb_bottom = 0
        # self.bb_top = 0

        self.move_y_grid = 0
        self.move_x_grid = 0

        self.x_world = 0
        self.y_world = 0

        self.is_active = True

        self.type = BlockType.random()
        self.rotation = 0
        self.rotation_pending = 0

        self.thresh_time_passed_input = kwargs.pop("thresh_time_passed_input", 0.03)
        self.thresh_time_passed_drop = kwargs.pop("thresh_time_passed_drop", 0.3)

        if BlockType.TypeL == self.type:
            self.array = self.L_array
        elif BlockType.TypeTriangle == self.type:
            self.array = self.Triangle_array
        elif BlockType.TypeS == self.type:
            self.array = self.S_array
        elif BlockType.TypeZ == self.type:
            self.array = self.Z_array
        elif BlockType.TypeBar == self.type:
            self.array = self.Bar_array
        elif BlockType.TypeBlock == self.type:
            self.array = self.Block_array
        elif BlockType.TypeLinv == self.type:
            self.array = self.L_inv_array
        else:
            raise TypeError("Unknown type {}".format(self.type))

        # self.array = np.array(self.array)

        # self.update_bounding_box()
        width = self.array[self.rotation].shape[0]
        height = self.array[self.rotation].shape[1]
        self.x_grid = np.random.randint(0, size_grid_x - width)
        self.y_grid = self.size_grid_y - height - 1

    def rotate(self) -> None:
        self.rotation_pending += 1

        if self.rotation_pending >= len(self.array):
            self.rotation_pending = 0

    def drop(self, array: np.array, delta_time: float) -> None:
        self.time_passed_drop += delta_time

        if self.time_passed_drop > self.thresh_time_passed_drop:
            self.time_passed_drop = 0

            if not self.is_blocked(array, self.x_grid, self.y_grid - 1, self.rotation):
                self.y_grid = self.coerce_y_grid(self.y_grid - 1)
            else:
                self.is_active = False

    def process_input(self, array_world: np.array, delta_time: float) -> None:
        self.time_passed_input += delta_time

        if self.time_passed_input > self.thresh_time_passed_input:
            self.time_passed_input = 0
            if not self.is_blocked(
                    array_world,
                    self.x_grid + self.move_x_grid,
                    self.y_grid,
                    self.rotation_pending,
            ):
                self.x_grid += self.move_x_grid
                self.rotation = self.rotation_pending

            self.move_x_grid = 0
            self.move_y_grid = 0

        self.x_grid = self.coerce_x_grid(self.x_grid)
        self.y_grid = self.coerce_y_grid(self.y_grid)

    def update(self, array: np.array, delta_time: float) -> np.array:
        if self.is_active:
            self.drop(array, delta_time)

            if not self.is_active:
                return self.update_grid(array)

        if self.is_active:
            self.process_input(array, delta_time)

            if not self.is_active:
                return self.update_grid(array)

        return array

    def update_grid(self, array: np.array, do_copy=False) -> np.array:
        """Write current block to world grid"""
        array_ = array

        if do_copy:
            array_ = array.copy()

        width = self.array[self.rotation].shape[0]
        height = self.array[self.rotation].shape[1]

        # try:
        array_[self.x_grid: self.x_grid + width, self.y_grid: self.y_grid + height] += self.array[self.rotation]
        # except:
        #     print("bam")

        return array_

    def is_blocked(self, array: np.array, x_grid: int, y_grid: int, rotation: int, ) -> bool:
        width = self.array[rotation].shape[0]
        height = self.array[rotation].shape[1]

        # check if in scope
        if x_grid < 0:
            return True

        if x_grid + width > array.shape[0]:
            return True

        if y_grid < 0:
            return True

        array_ = (
                array[x_grid: x_grid + width, y_grid: y_grid + height]
                + self.array[rotation]
        )

        return (array_ > 1).any().any()

    def coerce_y_grid(self, y_grid: int) -> int:
        if y_grid < 0:
            y_grid = 0
            self.is_active = False

        return y_grid

    def coerce_x_grid(self, x_grid: int) -> int:
        width = self.array[self.rotation].shape[0]

        return min(max(0, x_grid), self.size_grid_x - width)

    def draw(self) -> None:
        for x in range(len(self.array[self.rotation])):
            for y in range(len(self.array[self.rotation][0])):
                if self.array[self.rotation][x][y] != 0:
                    x_world = (x + self.x_grid) * self.image.width
                    y_world = (y + self.y_grid) * self.image.height
                    self.image.center_x = x_world + self.image.width / 2
                    self.image.center_y = y_world + self.image.height / 2
                    self.image.draw()

        self.x_world = int(self.x_grid * self.image.width)
        self.y_world = int(self.y_grid * self.image.height)

    def set_moving_vector(self, move_x_grid: int, move_y_grid: int, ) -> None:
        self.move_x_grid = move_x_grid
        self.move_y_grid = move_y_grid

    @property
    def x(self) -> int:
        return self.x_world

    @property
    def y(self) -> int:
        return self.y_world
