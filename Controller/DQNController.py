import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import keras

GAMMA = 0.9
LEARNING_RATE = 0.01

MEMORY_SIZE = 1000000
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.999


class DQNController:

    def __init__(self, observation_space, action_space, check_name: str, gpu_count: int = 1, cpu_count: int = 16):

        config = tf.ConfigProto(device_count={'GPU': gpu_count, 'CPU': cpu_count})

        sess = tf.Session(config=config)
        keras.backend.set_session(sess)

        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        # self.model = Sequential()
        # self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        # self.model.add(Dense(24, activation="relu"))
        # self.model.add(Dense(self.action_space, activation="linear"))
        # self.model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=LEARNING_RATE))
        self.model = Sequential()
        self.model.add(Dense(6, input_shape=(observation_space,), activation="tanh"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        self.checkpoint = keras.callbacks.ModelCheckpoint(filepath=f"./{check_name}.hdf5", verbose=1,
                                                          period=10000)
        self.callbacks_list = [self.checkpoint]
        self.parameter_history = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0, callbacks=self.callbacks_list)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

