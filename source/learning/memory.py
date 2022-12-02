from collections import deque
from typing import List, Tuple

import numpy as np


class Memory:
    def __init__(self, max_size: int) -> None:
        self.buffer = deque(maxlen=max_size)

    def add(self, experience: Tuple) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List:
        buffer_size = len(self.buffer)

        index = np.random.choice(
            np.arange(buffer_size),
            size=min(buffer_size, batch_size),
            replace=False,
        )

        return [self.buffer[i] for i in index]

    @property
    def empty(self) -> bool:
        return len(self.buffer) == 0
