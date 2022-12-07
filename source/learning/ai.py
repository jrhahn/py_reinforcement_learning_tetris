import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from data_types.game_state import GameState
from learning.memory import Memory

logger = logging.getLogger(__name__)


class AI:

    def __init__(
            self,
            state_size: Tuple,
            actions: List,
            learning_rate: float,
    ):
        self._state_size = state_size

        self._actions = actions
        self._num_actions = len(actions)

        self._q_model = self._build_model(
            optimizer=Adam(lr=learning_rate)
        )

        self._memory = Memory(max_size=10000)

        self._current_step = 0
        self._max_steps = 50000  # todo value?

        self._decay_step = 0
        self._decay_rate = 0.00001
        self._episode = 0
        self._gamma = 0.99

        self._path_save = Path("..") / "model"
        self._path_save.mkdir(exist_ok=True)

        self._current_state = AI._init_stack(state_size=state_size, )

    @staticmethod
    def _init_stack(state_size: Tuple) -> np.ndarray:
        return np.zeros(state_size)

    def _build_model(
            self,
            optimizer: OptimizerV2,
    ) -> Model:
        input_state = Input(shape=list(self._state_size))

        x = Conv2D(32, (3, 3), activation='relu')(input_state)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        # x = MaxPooling2D((2, 2))(x)
        # x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        output_state = Dense(len(self._actions), activation="linear")(x)

        input_action = Input(shape=[len(self._actions)])
        output_action = input_action

        q = tensorflow.math.multiply(output_state, output_action)

        model = Model(inputs=[input_state, input_action], outputs=q)
        model.compile(loss="mse", optimizer=optimizer)

        return model

    def save(self) -> None:
        self._q_model.save(self._path_save)
        logger.info(f"Model saved to {self._path_save}")

    @staticmethod
    def _stack_frames(
            frame: np.array,
            stacked_frames: np.ndarray,
            is_new_episode: bool,
    ) -> np.ndarray:
        if is_new_episode:
            stacked_frames = AI._init_stack(state_size=stacked_frames.shape)
        else:
            stacked_frames = np.dstack((stacked_frames[:, :, -1], frame))

        assert stacked_frames.shape[2] == 2, f"shape is {stacked_frames.shape}"

        return stacked_frames

    def predict_action(self) -> int:
        explore_probability = np.random.rand()

        threshold = 0.9  # np.exp(-self._decay_rate * self._decay_step)

        if explore_probability < threshold:
            logger.debug(
                f"Exploration:"
                f"(decay rate: {self._decay_rate}, decay_step: {self._decay_step})"
            )

            choice = np.random.randint(0, self._num_actions)
        else:
            logger.debug("Prediction")

            actions_cat = self.actions_to_one_hot(actions=self._actions)

            q_values = self._q_model.predict(
                (np.tile(self._current_state, [len(self._actions), 1, 1, 1]),
                 actions_cat)
            )

            # todo diag -> loss fun
            choice = np.argmax(np.diag(q_values))

        return self._actions[choice]

    def set_reward(
            self,
            action: int,
            state: np.array,
            game_state: GameState,
            reward: float,
    ) -> None:
        self._current_step -= 1

        next_state = self._stack_frames(
            frame=state,
            stacked_frames=self._current_state.copy(),
            is_new_episode=False,
        )

        if game_state == GameState.End or self._current_step == 0:
            frame_next = np.zeros(self._state_size[:-1], dtype=np.int)

            next_state = self._stack_frames(
                frame=frame_next,
                stacked_frames=self._current_state.copy(),
                is_new_episode=False,
            )

            self._current_step = self._max_steps

            self._episode += 1

            self._memory.add((next_state, action, reward, next_state, True))

        else:
            if self._current_state is None:
                self._current_state = next_state

            self._memory.add(
                (self._current_state, action, reward, next_state, False)
            )

        self._current_state = next_state

    def actions_to_one_hot(self, actions: np.array):
        actions_inv = {v: k for k, v in enumerate(self._actions)}

        identity_matrix = np.eye(len(self._actions))

        actions_cat = [identity_matrix[actions_inv[d], :] for d in actions]

        return np.vstack(actions_cat)

    def train(self) -> None:
        logger.info("Training")

        if self._memory.empty:
            return

        batch = self._memory.sample(batch_size=256)

        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb = np.array([each[4] for each in batch])

        target_qs_batch = []

        actions_cat = self.actions_to_one_hot(actions=actions_mb)

        q_next_state = self._q_model.predict(
            (next_states_mb, actions_cat)
        )

        if len(q_next_state) != next_states_mb.shape[0]:
            raise ValueError(f"Wrong size: {len(q_next_state)} (expected {next_states_mb.shape[0]})")

        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
        for i in range(0, len(batch)):
            if dones_mb[i]:
                target_qs_batch.append(rewards_mb[i])

            else:
                target = rewards_mb[i] + self._gamma * np.max(q_next_state[i])
                target_qs_batch.append(target)

        targets_mb = np.array(target_qs_batch)

        return self._q_model.fit(
            x=(states_mb, actions_cat),
            y=targets_mb,
            epochs=10,
        )
