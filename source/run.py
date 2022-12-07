#!/usr/bin/env python
import logging
import time

from tetris.arcade import ArcadeTetris
from tetris.no_display import NoDisplayTetris

logging.basicConfig(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)


def run(
        run_with_display: bool = False,
        screen_height: int = 480,
        screen_width: int = 320
) -> None:
    if run_with_display:
        game = ArcadeTetris(screen_width=screen_width, screen_height=screen_height, )
    else:
        game = NoDisplayTetris(screen_width=screen_width, screen_height=screen_height, )

    duration = 0
    while True:
        start_time = time.time()
        game.update(delta_time=duration)
        duration = time.time() - start_time

        logger.debug(f"Cycle time: {(1 / float(duration))}")


if __name__ == "__main__":
    run()
