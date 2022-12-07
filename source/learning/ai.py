import logging
from typing import List, Tuple

import numpy as np
import tensorflow
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Flatten
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

        # Initialize the decay rate (that will use to reduce epsilon)
        self._decay_step = 0

        # self.saver = tf.train.Saver()

        self._memory_size = 10000

        self._memory = Memory(max_size=self._memory_size)

        self.current_step = 0
        self.current_state = None
        self.max_steps = 50000  # todo value?

        self.episode_rewards = list()
        self.l_rewards = list()
        self.total_episodes = 50
        self.max_steps = 50000
        self.batch_size = 64
        self.decay_rate = 0.00001
        self.episode = 0
        self.gamma = 0.99

        self.num_frames_stacked = 2

        # Initialize deque with zero-images one array for each image
        self._current_state = AI._init_stack(state_size=state_size, )

    @staticmethod
    def _init_stack(state_size: Tuple) -> np.ndarray:
        # d = deque(maxlen=self.num_frames_stacked)
        d = np.zeros(state_size)
        # for ii in range(self.num_frames_stacked):
        #     d.append(np.zeros(state_size, dtype=np.int))

        return d

    def _build_model(
            self,
            optimizer: OptimizerV2,
            # name="AINetwork",
    ) -> Model:
        #
        # input_state = Input(shape=list(self._state_size) + [1])
        input_state = Input(shape=list(self._state_size))

        # output_action = Dense(1, activation="relu")(input_action)

        # output_action = tensorflow.reshape(input_action,
        #                                    (num_frames, 1, 1, -1))  # tensorflow.tile(input_action, (1,1,1)),
        # input_b= Input(shape=(num_frames, len(self._actions)))

        # the first branch operates on the first input
        x = Flatten()(input_state)
        x = Dense(8, activation="relu")(x)
        x = Dense(2, activation="relu")(x)
        # x = Model(inputs=input_a, outputs=x)

        # the second branch opreates on the second input
        # y = Dense(64, activation="relu")(input_b)
        # y = Dense(32, activation="relu")(y)
        # y = Dense(4, activation="relu")(y)
        # y = Model(inputs=input_b, outputs=y)
        # combine the output of the two branches
        # combined = concatenate([x.output, y.output])

        # apply a FC layer and then a regression prediction on the
        # combined outputs
        # z = Dense(2, activation="relu")(combined)
        # z = Dense(2, activation="relu")(x)
        output_state = Dense(2, activation="linear")(x)
        output_state = Dense(len(self._actions), activation="linear")(output_state)
        # output_state = tile(x, [1, 1, 1, len(self._actions)])

        # input_action = Input(shape=[1, 1, 1, len(self._actions)])
        # output_action = tile(input_action, [1] + list(self._state_size) + [1])
        input_action = Input(shape=[len(self._actions)])
        output_action = input_action
        # output_action = Dense(len(self._actions), activation="linear")(input_action)
        # output_action = input_action
        # output_action = tile(input_action, [1] + list(self._state_size) + [1])

        # q = tensorflow.math.reduce_sum(tensorflow.math.multiply(output_state, output_action))
        q = tensorflow.math.multiply(output_state, output_action)

        Q = Model(inputs=[input_state, input_action], outputs=q)
        # Q = Model(inputs=input_state, outputs=q)
        # Q = Model(inputs= input_action, outputs=output_action)
        Q.compile(loss="mse", optimizer=optimizer)

        # our model will accept the inputs of the two branches and
        # then output a single value

        # model = Sequential()
        # # model.add(Embedding(input_dim=np.prod(self._state_size) * num_frames, output_dim=10, input_length=1))
        # model.add(Dense(input_shape=list(self._state_size) + [num_frames], units=200, activation="relu"))
        # # model.add(Reshape((10,)))
        # model.add(Dense(50, activation="relu"))
        # model.add(Dense(50, activation="relu"))
        # model.add(Dense(len(self._actions), activation="linear"))

        # z.compile(loss="mse", optimizer=optimizer)
        # Q is our predicted Q value.
        # actions = placeholder(int, [None, int(len(self._actions))], name="actions_")
        # Q = tensorflow.math.reduce_sum(tensorflow.math.multiply(output, actions))

        # The loss is the difference between our predicted Q_values and the Q_target
        # Sum(Qtarget - Q)^2

        return Q

    def save(self):
        # save_path = self.saver.save(self.sess, "./models/model.ckpt")
        # print("Model saved to {}".format(save_path))
        pass

    #     # Add ops to save and restore all the variables.
    #     saver = tf.train.Saver()
    #
    #     # Later, launch the model, initialize the variables, do some work, and save the
    #     # variables to disk.
    #     with tf.Session() as sess:
    #         sess.run(init_op)
    #         # Do some work with the model.
    #         inc_v1.op.run()
    #         dec_v2.op.run()
    #         # Save the variables to disk.
    #         save_path = saver.save(sess, "/tmp/model.ckpt")
    #         print("Model saved in path: %s" % save_path)

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

    # def predict_action(self, explore_start, explore_stop, state, actions):
    def predict_action(self, state: np.array) -> int:
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        # explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        explore_probability = np.exp(-self.decay_rate * self._decay_step)

        if explore_probability > exp_exp_tradeoff:
            logger.info(
                f"Exploration: {explore_probability} > {exp_exp_tradeoff}"
                f"(decay rate: {self.decay_rate}, decay_step: {self._decay_step})"
            )
            # Make a random action (exploration)
            choice = np.random.randint(0, self._num_actions)
        else:
            logger.info("Prediction")
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # todo try all actions

            actions_cat = self.actions_to_one_hot(actions=self._actions)

            q_values = self._q_model.predict(
                (np.tile(self._current_state, [len(self._actions), 1, 1, 1]),
                 actions_cat)
            )

            # Take the biggest Q value (= the best action)
            choice = np.argmax(q_values)

        current_action = self._actions[choice]

        return current_action  # , explore_probability

    def set_reward(
            self,
            action: int,
            state: np.array,
            game_state: GameState,
            reward: float,
    ) -> None:
        # Add the reward to total reward
        self.episode_rewards.append(reward)

        self.current_step -= 1

        next_state = self._stack_frames(
            frame=state,
            stacked_frames=self._current_state.copy(),
            is_new_episode=False,
        )

        # If the game is finished
        if game_state == GameState.End or self.current_step == 0:

            logger.info("ADD MEMORY")
            # The episode ends so no next state
            frame_next = np.zeros(self._state_size[:-1], dtype=np.int)

            next_state = self._stack_frames(
                frame=frame_next,
                stacked_frames=self._current_state.copy(),
                is_new_episode=False,
            )

            # Set step = max_steps to end the episode
            self.current_step = self.max_steps

            self.episode += 1

            # Get the total reward of the episode
            total_reward = np.sum(self.episode_rewards)

            # print('Episode: {}'.format(episode),
            #       'Total reward: {}'.format(total_reward),
            #       'Explore P: {:.4f}'.format(explore_probability),
            #       'Training Loss {:.4f}'.format(loss))

            self.l_rewards.append((self.episode, total_reward))

            # Store transition <st,at,rt+1,st+1> in memory D
            self._memory.add((next_state, action, reward, next_state, True))

        else:
            # Stack the frame of the next_state
            # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            if self._current_state is None:
                self._current_state = next_state

            # Add experience to memory
            self._memory.add(
                (self._current_state, action, reward, next_state, False)
            )

        self._current_state = next_state

    def actions_to_one_hot(self, actions: np.array):
        actions_inv = {v: k for k, v in enumerate(self._actions)}

        identiy_matrix = np.eye(len(self._actions))

        actions_cat = [identiy_matrix[actions_inv[d], :] for d in actions]

        return np.vstack(actions_cat)

    def train(self) -> None:
        logger.info("Training")

        if self._memory.empty:
            return

        batch = self._memory.sample(self.batch_size)

        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb = np.array([each[4] for each in batch])

        target_qs_batch = []

        actions_cat = self.actions_to_one_hot(actions=actions_mb)

        # Get Q values for next_state
        q_next_state = self._q_model.predict(
            (next_states_mb, actions_cat)
        )

        if len(q_next_state) != next_states_mb.shape[0]:
            raise ValueError(f"Wrong size: {len(q_next_state)} (expected {next_states_mb.shape[0]})")

        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
        for i in range(0, len(batch)):
            # If we are in a terminal state, only equals reward
            if dones_mb[i]:
                target_qs_batch.append(rewards_mb[i])

            else:
                target = rewards_mb[i] + self.gamma * np.max(q_next_state[i])
                target_qs_batch.append(target)

        # targets_mb = np.array([each for each in target_qs_batch])
        targets_mb = np.array(target_qs_batch)

        # convert actions

        if len(states_mb) != len(targets_mb):
            raise ValueError("bla")

        history = self._q_model.fit(
            x=(states_mb, actions_cat),
            y=targets_mb
        )

        print()

        # loss, _ = self.sess.run([self.loss, self.optimizer],
        #                         feed_dict={self.inputs: states_mb,
        #                                    self.target_Q: targets_mb,
        #                                    self.actions: l_actions_cat})

        # # Write TF Summaries
        # summary = sess.run(self.write_op, feed_dict={self.inputs_: states_mb,
        #                                              self.target_Q: targets_mb,
        #                                              self.actions_: actions_mb})
        # self.writer.add_summary(summary, self.episode)
        # self.writer.flush()

        # Save model every 5 episodes

    # def test(self):
    #     with tf.Session() as sess:
    #         total_test_rewards = []
    #
    #         # Load the model
    #         saver.restore(sess, "./models/model.ckpt")
    #
    #         for episode in range(1):
    #             total_rewards = 0
    #
    #             state = env.reset()
    #             state, stacked_frames = stack_frames(stacked_frames, state, True)
    #
    #             print("****************************************************")
    #             print("EPISODE ", episode)
    #
    #             while True:
    #                 # Reshape the state
    #                 state = state.reshape((1, *state_size))
    #                 # Get action from Q-network
    #                 # Estimate the Qs values state
    #                 Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state})
    #
    #                 # Take the biggest Q value (= the best action)
    #                 choice = np.argmax(Qs)
    #                 action = possible_actions[choice]
    #
    #                 # Perform the action and get the next_state, reward, and done information
    #                 next_state, reward, done, _ = env.step(action)
    #                 env.render()
    #
    #                 total_rewards += reward
    #
    #                 if done:
    #                     print("Score", total_rewards)
    #                     total_test_rewards.append(total_rewards)
    #                     break
    #
    #                 next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    #                 state = next_state
    #
    #         env.close()

# Instantiate the DQNetwork
# DQNetwork = DQNetwork(state_size, action_size, learning_rate)

# class AI(object):
#
#     def __init__(self):
#         self.model = self.build_model()
#
#         self.num_actions = 3
#         self.learning_rate = 0.001
#
#     def build_model(self):
#

# def build_model(self):
#     model = Sequential()
#     model.add(Conv2D(32, input_shape=(84, 84, 4), kernel_size=[8, 8], strides=[4, 4], padding="VALID"))
#     model.add(ELU())
#     model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID"))
#     model.add(ELU())
#     model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="VALID"))
#     model.add(ELU())
#     model.add(Flatten())
#     model.add(Dense(512, activation=advanced_activations.ELU))
#     model.add(Dense(self.num_actions))
#
#     def custom_objective(y_true, y_pred):
#         '''Just another crossentropy'''
#         y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
#         y_pred /= y_pred.sum(axis=-1, keepdims=True)
#         cce = T.nnet.categorical_crossentropy(y_pred, y_true)
#         return cce
#
#     def loss_fun():
#         # Q is our predicted Q value.
#         self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
#
#         # The loss is the difference between our predicted Q_values and the Q_target
#         # Sum(Qtarget - Q)^2
#         self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
#
#     model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate),
#                   loss=loss_fun)
#
# def get_action(self, current_block, array):
#     2
