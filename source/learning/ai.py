import logging
from collections import deque
from typing import List, Tuple, Deque

import numpy as np
from tensorflow.python.keras.layers import Embedding, Reshape, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from data_types.game_state import GameState
from learning.memory import Memory
from tetris_manager import TetrisManager

logger = logging.getLogger(__name__)


class AI:

    def __init__(self, state_size: Tuple, actions: List, learning_rate: float, name: str = "DQNetwork", ):
        self._state_size = state_size

        self._actions = actions
        self._num_actions = len(actions)
        # self._earning_rate = learning_rate

        self._build_model(
            optimizer=Adam(lr=learning_rate)
        )

        # todo here
        # self.sess = tf.Session()
        # Initialize the variables
        # self.sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        self._decay_step = 0

        # self.saver = tf.train.Saver()

        self._memory_size = 10000

        self._memory = Memory(max_size=self._memory_size)

        self.current_step = 0
        self.current_action = 0
        self.current_state = None
        self.max_steps = 50000  # todo value?

        self.l_episode_rewards = list()
        self.l_rewards = list()
        self.total_episodes = 50
        self.max_steps = 50000
        self.batch_size = 64
        self.decay_rate = 0.00001
        self.episode = 0
        self.gamma = 0.99

        self.num_frames_stacked = 2

        self._prev_state = None

        # Initialize deque with zero-images one array for each image
        self.stacked_frames = self._init_stack(state_size=state_size, )

    def _init_stack(self, state_size: Tuple) -> Deque:
        return deque(
            [np.zeros(state_size, dtype=np.int)] * self.num_frames_stacked,
            maxlen=2,
        )

    def _build_model(self, optimizer: OptimizerV2, name="AINetwork"):

        model = Sequential()
        model.add(Embedding(input_dim=np.prod(self._state_size), output_dim=10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(len(self._actions), activation="linear"))

        model.compile(loss="mse", optimizer=optimizer)
        return model

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

    def _stack_frames(self, state: np.array, is_new_episode: bool, ) -> np.array:
        # Preprocess frame
        # frame = preprocess_frame(state)
        frame = state

        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_frames = self._init_stack(state_size=self._state_size)

            # Because we're in a new episode, copy the same frame 4x
            for ii in range(self.num_frames_stacked):
                self.stacked_frames.append(frame)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(self.stacked_frames, axis=2)

        return stacked_state
        # return self.stacked_frames

    # def predict_action(self, explore_start, explore_stop, state, actions):
    def predict_action(self, state):
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        # explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        explore_probability = np.exp(-self.decay_rate * self._decay_step)

        if explore_probability > exp_exp_tradeoff:
            # Make a random action (exploration)
            choice = np.random.randint(0, self._num_actions)
        else:
            logger.info("Prediction")
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Qs = self.sess.run(self.output, feed_dict={self.inputs: state.reshape((1, *state.shape))})
            q_values = self._q_network.predict(state)

            # Take the biggest Q value (= the best action)
            choice = np.argmax(q_values)

        self.current_action = self._actions[choice]

        return self.current_action  # , explore_probability

    def get_action(self, tm: TetrisManager) -> int:
        # Increase decay_step
        self._decay_step += 0.001

        # Predict the action to take and take it
        # self.current_action, explore_probability = self.predict_action(tm.n_array)

        current = tm.state
        if self._prev_state is None:
            self._prev_state = current

        state = np.stack((self._prev_state, current), axis=2)
        self._prev_state = current

        self.current_state = self.stack_frames(state, True)
        self.current_action = self.predict_action(self.current_state)

        return self.current_action

    def set_reward(self, tm: TetrisManager, game_state: GameState, reward: float) -> None:
        # Add the reward to total reward
        self.l_episode_rewards.append(reward)

        self.current_step -= 1

        current = tm.state
        if self._prev_state is None:
            self._prev_state = current

        state = np.stack((self._prev_state, current), axis=2)
        self._prev_state = current

        next_state = self._stack_frames(state, False)

        # If the game is finished
        if game_state == GameState.End or self.current_step == 0:

            logger.info("ADD MEMORY")
            # The episode ends so no next state
            next_state = np.zeros(self._state_size[:-1], dtype=np.int)

            next_state = self._stack_frames(next_state, False)

            # Set step = max_steps to end the episode
            self.current_step = self.max_steps

            self.episode += 1

            # Get the total reward of the episode
            total_reward = np.sum(self.l_episode_rewards)

            # print('Episode: {}'.format(episode),
            #       'Total reward: {}'.format(total_reward),
            #       'Explore P: {:.4f}'.format(explore_probability),
            #       'Training Loss {:.4f}'.format(loss))

            self.l_rewards.append((self.episode, total_reward))

            # Store transition <st,at,rt+1,st+1> in memory D
            self._memory.add((next_state, self.current_action, reward, next_state, True))

        else:
            # Stack the frame of the next_state
            # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            # Add experience to memory
            self._memory.add(
                (self.current_state, self.current_action, reward, next_state, False)
            )

            # st+1 is now our current state
            # state = next_state

    def get_block_state(self, tm: TetrisManager) -> Tuple:
        return tm.block_active.type, tm.block_active.x_grid, tm.block_active.y_grid

    def actions_to_one_hot(self, l_data):
        d_actions_inv = {k: v for v, k in enumerate(self._actions)}

        n_vecs = np.eye(len(self._actions))

        l_actions_cat = [n_vecs[d_actions_inv[d], :] for d in l_data]

        return np.vstack(l_actions_cat)

    def train(self) -> None:
        logger.info("Training")

        # todo add memory
        # Obtain random mini-batch from memory

        if self._memory.empty:
            return

        batch = self._memory.sample(self.batch_size)

        return
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        # dones_mb = np.array([each[3] for each in batch])
        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []

        # Get Q values for next_state
        Qs_next_state = self.sess.run(self.output, feed_dict={self.inputs_: next_states_mb})
        # Qs_next_state = self.sess.run(self.output, feed_dict={self.inputs: states_mb})

        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
        for i in range(0, len(batch)):
            terminal = dones_mb[i]

            # If we are in a terminal state, only equals reward
            if terminal:
                target_Qs_batch.append(rewards_mb[i])

            else:
                target = rewards_mb[i] + self.gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)

        targets_mb = np.array([each for each in target_Qs_batch])

        # convert actions

        l_actions_cat = self.actions_to_one_hot(actions_mb)

        loss, _ = self.sess.run([self.loss, self.optimizer],
                                feed_dict={self.inputs: states_mb,
                                           self.target_Q: targets_mb,
                                           self.actions: l_actions_cat})

        # Write TF Summaries
        summary = sess.run(self.write_op, feed_dict={self.inputs_: states_mb,
                                                     self.target_Q: targets_mb,
                                                     self.actions_: actions_mb})
        self.writer.add_summary(summary, self.episode)
        self.writer.flush()

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
