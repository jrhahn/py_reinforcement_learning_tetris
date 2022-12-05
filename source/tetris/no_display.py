import logging

from ai import AI
from data_types.game_state import GameState
from tetris.arcade import ArcadeTetris
from tetris_manager import TetrisManager

logger = logging.getLogger(__name__)


class NoDisplayTetris(object):
    arcade_key_left = "akleft"
    arcade_key_right = "akright"
    arcade_key_up = "akup"

    actions = [arcade_key_left, arcade_key_right, arcade_key_up]

    def __init__(self, screen_width: int, screen_height: int, ):

        self._do_use_player_control = False

        self._thresh_time_passed_input = 0
        self._thresh_time_passed_drop = 0

        self._screen_width = screen_width
        self._screen_height = screen_height

        self._tm = TetrisManager(
            image=None,
            screen_width=screen_width,
            screen_height=screen_height,
            thresh_time_passed_input=0,
            thresh_time_passed_drop=0.001,
        )

        self._cnt_save = 0

        self._ai = AI(
            state_size=(self._tm.size_grid_x, self._tm.size_grid_y, 2),
            actions=ArcadeTetris.l_actions,
            learning_rate=0.0001,
        )

    def update(self, delta_time: float) -> None:
        game_state = self._tm.update(delta_time)

        if not self._do_use_player_control:
            key = self._ai.predict_action(self._tm.state)

            if key == NoDisplayTetris.arcade_key_left:
                self._tm.set_move(move_x=-1, move_y=0)
            elif key == NoDisplayTetris.arcade_key_right:
                self._tm.set_move(move_x=1, move_y=0)
            elif key == NoDisplayTetris.arcade_key_up:
                self._tm.rotate()

            self._ai.set_reward(self._tm, game_state, self._tm.score)

        if game_state == GameState.End:
            self._ai.train()

            logger.info("Score: {}".format(self._tm.score))

            if self._cnt_save == 0:
                self._ai.save()
                self._cnt_save = 20

            self._cnt_save -= 1
            self._tm = TetrisManager(
                image=None,
                screen_width=self._screen_width,
                screen_height=self._screen_height,
                thresh_time_passed_input=self._thresh_time_passed_input,
                thresh_time_passed_drop=self._thresh_time_passed_drop,
            )

            # debug only
            self._ai.set_reward(self._tm, game_state, self._tm.score)
