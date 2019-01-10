import logging
import random

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
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1]
        ],
        [
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, 0]
        ],
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 0]
        ]]
    L_inv_array = [
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 0]
        ],
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 1]
        ]
    ]
    Triangle_array = [
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0]
        ]
    ]
    Block_array = [[
        [1, 1],
        [1, 1]]]
    Bar_array = [
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0]
        ],
        [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0]
        ],
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ]
    ]
    Z_array = [
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 1]
        ],
        [
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0]
        ]
    ]
    S_array = [
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 1, 0]
        ],
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ]
    ]

    def __init__(self, image, size_grid_x, size_grid_y, **kwargs):
        self.time_passed_input = 0
        self.time_passed_drop = 0
        self.image = image
        self.screen_height = size_grid_y * image.height

        self.size_grid_x = size_grid_x
        self.size_grid_y = size_grid_y

        self.bb_left = 0
        self.bb_right = 0
        self.bb_bottom = 0
        self.bb_top = 0

        self.move_y_grid = 0
        self.move_x_grid = 0

        self.x_world = 0
        self.y_world = 0

        self.is_active = True

        self.type = BlockType.random()
        self.current_rotation = 0

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

        self.array = np.array(self.array)

        self.update_bounding_box()
        width = self.bb_right - self.bb_left + 1
        self.x_grid = np.random.randint(-self.bb_left, size_grid_x - width)
        self.y_grid = self.size_grid_y - self.bb_bottom - 1

    def rotate(self):
        self.current_rotation += 1

        if self.current_rotation >= self.array.shape[0]:
            self.current_rotation = 0

        self.update_bounding_box()

    def update_bounding_box(self):
        x_min = np.inf
        x_max = 0

        y_min = np.inf
        y_max = 0

        for x in range(len(self.array[self.current_rotation])):
            for y in range(len(self.array[self.current_rotation][0])):
                if self.array[self.current_rotation][x][y] == 1:
                    x_min = min(x, x_min)
                    x_max = max(x, x_max)
                    y_min = min(y, y_min)
                    y_max = max(y, y_max)

        self.bb_left = y_min
        self.bb_right = y_max
        self.bb_top = x_min
        self.bb_bottom = x_max

        # self.bb_left = x_min
        # self.bb_bottom = y_max
        # self.bb_top = y_min
        # self.bb_right = x_max

    def drop(self, array, delta_time):
        self.time_passed_drop += delta_time

        # Log.i("", "attempt to drop block type: " + type + " " + delta_time)
        if self.time_passed_drop > self.thresh_time_passed_drop:
            self.time_passed_drop = 0

            if not self.is_blocked(array, self.x_grid, self.y_grid - 1):
                # self.y_grid -= 1
                self.y_grid = self.check_y_grid(self.y_grid - 1)
            else:
                self.is_active = False

    def process_input(self, array_world, delta_time):
        self.time_passed_input += delta_time

        if self.time_passed_input > self.thresh_time_passed_input:
            self.time_passed_input = 0
            if not self.is_blocked(array_world, self.x_grid + self.move_x_grid, self.y_grid):
                self.x_grid += self.move_x_grid

            # if not self.is_blocked(array_world, self.x_grid, self.y_grid + self.move_y_grid):
            #     self.y_grid += self.move_y_grid

            self.move_x_grid = 0
            self.move_y_grid = 0

            self.x_grid = self.check_x_grid(self.x_grid)
            self.y_grid = self.check_y_grid(self.y_grid)

    def update(self, array, delta_time):
        if self.is_active:
            is_already_blocked = self.is_blocked(array, self.x_grid, self.y_grid)

            self.drop(array, delta_time)

            if not self.is_active:
                return self.update_grid(array, is_already_blocked)

        if self.is_active:
            self.process_input(array, delta_time)

            if not self.is_active:
                return self.update_grid(array, is_already_blocked)

        return array

    def update_grid(self, array, is_already_blocked=False, do_copy=False):
        array_ = array

        if do_copy:
            array_ = array.copy()

        for x in range(self.bb_left, self.bb_right + 1):
            for y in range(self.bb_top, self.bb_bottom + 1):
                if self.array[self.current_rotation][x][y] == 1:
                    # try:
                    array_[x + self.x_grid][y + self.y_grid] += self.array[self.current_rotation][x][y]
                    # except IndexError:
                    #     print("bam")

                    # cell is covered by two items (can happen at the top -> game over)
                    if array_[x + self.x_grid][y + self.y_grid] > 1 and not is_already_blocked:
                        print("SHOULD NOT HAPPEN @ y: {}".format(y + self.y_grid))
        return array

    def is_blocked(self, array, x_grid, y_grid):
        width =  self.bb_right + 1 - self.bb_left
        for x in range(self.bb_left, self.bb_right + 1):
            for y in range(self.bb_top, self.bb_bottom + 1):
                if self.array[self.current_rotation][x][y] == 1:
                    x_offset = x + x_grid - self.bb_left
                    if (x_offset < -self.bb_left) or (x_offset + width >= self.size_grid_x):
                        return True

                    val = array[x + x_grid][y + y_grid] + self.array[self.current_rotation][x][y]

                    if val > 1:
                        return True

        return False

    def check_y_grid(self, y_grid):
        if y_grid <= -self.bb_top:
            # if y_grid > self.bb_bottom
            print("b:{}".format(self.bb_bottom))
            y_grid = -self.bb_top
            self.is_active = False
        # if y_grid <= len(self.array[self.current_rotation][0])-self.bb_bottom:
        #     y_grid = -(len(self.array[self.current_rotation][0])-self.bb_bottom)
        #     self.is_active = False

        return y_grid

    def check_x_grid(self, x_grid):
        x_offset = x_grid + self.bb_left
        if x_offset < -self.bb_left:
            x_grid = -self.bb_left
        elif x_grid + self.bb_right >= self.size_grid_x:
            x_grid = self.size_grid_x - self.bb_right - 1

        return x_grid

    def draw(self):
        for x in range(len(self.array[self.current_rotation])):
            for y in range(len(self.array[self.current_rotation][0])):
                if self.array[self.current_rotation][x][y] != 0:
                    x_world = (x + self.x_grid) * self.image.width
                    y_world = (y + self.y_grid) * self.image.height
                    self.image.center_x = x_world + self.image.width / 2
                    self.image.center_y = y_world + self.image.height / 2
                    self.image.draw()

        self.x_world = int((self.x_grid + (self.bb_right - self.bb_left) / 2.0) * self.image.width)
        self.y_world = int((self.y_grid + (self.bb_bottom - self.bb_top) / 2.0) * self.image.height)

    def setMovingVector(self, move_x_grid, move_y_grid):
        self.move_x_grid = move_x_grid
        self.move_y_grid = move_y_grid

    def get_x(self):
        return self.x_world

    def get_y(self):
        return self.y_world
