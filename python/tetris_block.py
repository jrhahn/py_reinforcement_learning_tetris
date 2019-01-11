import logging
import random

import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__name__)
import time
import numpy as np


class BlockType:
    TypeL = 'typel'
    TypeLinv = 'typelinv'
    TypeBlock = 'typeblock'
    TypeTriangle = 'typetriangle'
    TypeBar = 'typebar'
    TypeZ = 'typez'
    TypeS = 'types'

    l_types = [TypeL, TypeBar, TypeBlock, TypeLinv, TypeS, TypeZ, TypeTriangle]

    @staticmethod
    def random():
        return BlockType.l_types[random.randint(0, len(BlockType.l_types) - 1)]


class TetrisBlock:
    L_array = [
        np.array([
            [1, 0],
            [1, 0],
            [1, 1]
        ]),
        np.array([
            [1, 1, 1],
            [1, 0, 0]
        ]),
        np.array([
            [1, 1],
            [0, 1],
            [0, 1]
        ]),
        np.array([
            [0, 0, 1],
            [1, 1, 1],
        ])]
    L_inv_array = [
        np.array([
            [0, 1],
            [0, 1],
            [1, 1]
        ]),
        np.array([
            [1, 0, 0],
            [1, 1, 1],
        ]),
        np.array([
            [1, 1],
            [1, 0],
            [1, 0]
        ]),
        np.array([
            [1, 1, 1],
            [0, 0, 1]
        ])
    ]
    Triangle_array = [
        np.array([
            [0, 1, 0],
            [1, 1, 1],
        ]),
        np.array([
            [1, 0],
            [1, 1],
            [1, 0]
        ]),
        np.array([
            [1, 1, 1],
            [0, 1, 0]
        ]),
        np.array([
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0]
        ])
    ]
    Block_array = [
        np.array([
            [1, 1],
            [1, 1]])
    ]
    Bar_array = [
        np.array([
            [1, 1, 1, 1]
        ]),
        np.array([
            [1],
            [1],
            [1],
            [1]
        ]),
    ]
    Z_array = [
        np.array([
            [1, 1, 0],
            [0, 1, 1]
        ]),
        np.array([
            [0, 1],
            [1, 1],
            [1, 0]
        ])
    ]
    S_array = [
        np.array([
            [0, 1, 1],
            [1, 1, 0]
        ]),
        np.array([
            [1, 0],
            [1, 1],
            [0, 1]
        ])
    ]

    def __init__(self, image, size_grid_x, size_grid_y, **kwargs):
        self.time_passed_input = 0
        self.time_passed_drop = 0
        self.image = image
        self.screen_height = size_grid_y * image.height

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

        self.thresh_time_passed_input = kwargs.pop('thresh_time_passed_input', 0.03)
        self.thresh_time_passed_drop = kwargs.pop('thresh_time_passed_drop', .3)

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

    def rotate(self):
        self.rotation_pending += 1

        if self.rotation_pending >= len(self.array):
            self.rotation_pending = 0

    def drop(self, array, delta_time):
        self.time_passed_drop += delta_time

        if self.time_passed_drop > self.thresh_time_passed_drop:
            self.time_passed_drop = 0

            if not self.is_blocked(array, self.x_grid, self.y_grid - 1, self.rotation):
                self.y_grid = self.check_y_grid(self.y_grid - 1)
            else:
                self.is_active = False

    def process_input(self, array_world, delta_time):
        self.time_passed_input += delta_time

        if self.time_passed_input > self.thresh_time_passed_input:
            self.time_passed_input = 0
            if not self.is_blocked(array_world, self.x_grid + self.move_x_grid, self.y_grid, self.rotation_pending):
                self.x_grid += self.move_x_grid
                self.rotation = self.rotation_pending

            self.move_x_grid = 0
            self.move_y_grid = 0

        self.x_grid = self.check_x_grid(self.x_grid)
        self.y_grid = self.check_y_grid(self.y_grid)

    def update(self, array, delta_time):
        if self.is_active:
            self.drop(array, delta_time)

            if not self.is_active:
                return self.update_grid(array)

        if self.is_active:
            self.process_input(array, delta_time)

            if not self.is_active:
                return self.update_grid(array)

        return array

    def update_grid(self, array, do_copy=False):
        """Write current block to world grid"""
        array_ = array

        if do_copy:
            array_ = array.copy()

        width = self.array[self.rotation].shape[0]
        height = self.array[self.rotation].shape[1]

        # try:
        array_[self.x_grid:self.x_grid + width, self.y_grid:self.y_grid + height] += self.array[self.rotation]
        # except:
        #     print("bam")

        return array_

    def is_blocked(self, array, x_grid, y_grid, rotation):
        width = self.array[rotation].shape[0]
        height = self.array[rotation].shape[1]

        # check if in scope
        if x_grid < 0:
            return True

        if x_grid + width > array.shape[0]:
            return True

        if y_grid < 0:
            return True

        array_ = array[x_grid:x_grid + width, y_grid:y_grid + height] + self.array[rotation]

        return (array_ > 1).any().any()

    def check_y_grid(self, y_grid):
        if y_grid < 0:
            y_grid = 0
            self.is_active = False

        return y_grid

    def check_x_grid(self, x_grid):
        width = self.array[self.rotation].shape[0]

        if x_grid < 0:
            x_grid = 0
        elif x_grid + width > self.size_grid_x:
            x_grid = self.size_grid_x - width

        return x_grid

    def draw(self):
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

    def setMovingVector(self, move_x_grid, move_y_grid):
        self.move_x_grid = move_x_grid
        self.move_y_grid = move_y_grid

    def get_x(self):
        return self.x_world

    def get_y(self):
        return self.y_world
