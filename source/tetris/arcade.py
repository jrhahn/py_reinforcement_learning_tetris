import arcade
import numpy as np

from data_types.game_state import GameState
from learning.ai import AI
from tetris_manager import TetrisManager


class ArcadeTetris(arcade.Window):
    actions = [arcade.key.LEFT, arcade.key.RIGHT, arcade.key.UP, None]

    def __init__(self, screen_width: int, screen_height: int, ):

        self._do_use_player_control = False
        self._do_draw = True

        super().__init__(width=screen_width, height=screen_height, )
        # arcade.open_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Tetris")
        arcade.set_background_color(arcade.color.WHITE)

        self.set_update_rate(1 / 1000.0)

        self._image = arcade.Sprite("../resources/block.png", 0.25)

        self.do_use_player_control = False

        self._thresh_time_passed_input = 0
        self._thresh_time_passed_drop = 0

        self._screen_width = screen_width
        self._screen_height = screen_height

        if self.do_use_player_control:
            self._thresh_time_passed_input = 0.03
            self._thresh_time_passed_drop = 0.3

        self._tm = TetrisManager(
            image=self._image,
            screen_width=screen_width,
            screen_height=screen_height,
            thresh_time_passed_input=self._thresh_time_passed_input,
            thresh_time_passed_drop=self._thresh_time_passed_drop,
        )

        self._ai = AI(
            state_size=(self._tm.size_grid_x, self._tm.size_grid_y, 2),
            actions=ArcadeTetris.actions,
            learning_rate=0.0001,
        )

        self._prev_state = None

    def on_draw(self):
        if self._do_draw:
            arcade.start_render()
            self._tm.draw()
            arcade.finish_render()

    def on_key_press(self, key: int, modifiers: int, ):
        """Called whenever a key is pressed."""

        if self._do_use_player_control:
            if key == arcade.key.LEFT:
                self._tm.set_move(-1, 0)
            elif key == arcade.key.RIGHT:
                self._tm.set_move(1, 0)
            elif key == arcade.key.UP:
                self._tm.rotate()

    def update(self, delta_time: float):
        game_state = self._tm.update(delta_time)

        if not self.do_use_player_control:
            action = self._ai.predict_action(self._tm.state)
        else:
            action = 0  # todo get user input

        if action == arcade.key.LEFT:
            self._tm.set_move(-1, 0)
        elif action == arcade.key.RIGHT:
            self._tm.set_move(1, 0)
        elif action == arcade.key.UP:
            self._tm.rotate()

        self._ai.set_reward(
            action=action,
            state=self._tm.state,
            game_state=game_state,
            reward=self._tm.score,
        )

        if self._prev_state is None:
            self._prev_state = self._tm.state

        # self._ai.stack_frames(
        #     state=np.stack(
        #         (self._prev_state,
        #          self._tm.state), axis=2
        #     ),
        #     is_new_episode=game_state == GameState.Start
        # )

        self._prev_state = self._tm.state

        if game_state == GameState.End:
            self._ai.train()
            # self._ai.save()
            self._tm = TetrisManager(
                image=self._image,
                screen_width=self._screen_width,
                screen_height=self._screen_height,
                thresh_time_passed_input=self._thresh_time_passed_input,
                thresh_time_passed_drop=self._thresh_time_passed_drop,
            )
