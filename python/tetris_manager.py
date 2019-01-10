import numpy as np
from tetris_block import TetrisBlock


class TetrisManager:
    GAME_STATE_OK = 'game_state_ok'
    GAME_STATE_END = 'game_state_end'

    def __init__(self, image, display_x, display_y, **kwargs):
        self.image = image
        self.size_grid_x = int(display_x / image.width)
        self.size_grid_y = int(display_y / image.height)
        self.screen_height = self.size_grid_y * image.height

        self.n_array = np.zeros((self.size_grid_x, self.size_grid_y))

        self.block_active = None
        self.score = 0

        self.tetris_block_parameters = kwargs

        self.create_new_block()

    def get_state(self):
        return self.block_active.update_grid(self.n_array, do_copy=True)

    def create_new_block(self):
        print("Creating new block")
        self.block_active = TetrisBlock(self.image, self.size_grid_x, self.size_grid_y, **self.tetris_block_parameters)

    def draw(self):
        self.block_active.draw();

        for x in range(self.size_grid_x):
            for y in range(self.size_grid_y):
                if self.n_array[x, y] == 1:
                    self.image.center_x = (x + .5) * self.image.width
                    self.image.center_y = (y + .5) * self.image.height
                    self.image.draw()
                    # canvas.drawBitmap(self.image, x * self.image.getWidth(), y * self.image.getHeight(), None)

    def set_move(self, move_x, move_y):
        self.block_active.setMovingVector(move_x, move_y)

    def rotate(self):
        self.block_active.rotate()

    def get_block_size(self):
        return self.image.getWidth()

    def get_active_pos_x(self):
        return self.block_active.get_x()

    def get_active_pos_y(self):
        return self.block_active.get_y()

    def check_if_line_is_full(self):
        for y in range(self.size_grid_y - 1, -1, -1):
            is_complete = True

            for x in range(self.n_array.shape[0]):
                is_complete = (self.n_array[x, y] > 0) and is_complete

            if is_complete:
                self.score += 1
                for y_ in range(y, self.size_grid_y - 1):
                    for x in range(self.n_array.shape[0]):
                        self.n_array[x, y_] = self.n_array[x, y_ + 1]

    def update(self, delta_time):
        self.n_array = self.block_active.update(self.n_array, delta_time)

        if not self.block_active.is_active:
            print("is down")
            self.create_new_block()

            self.check_if_line_is_full()

        if self.n_array[:, -1].sum() > 0:
            return TetrisManager.GAME_STATE_END

        return TetrisManager.GAME_STATE_OK
