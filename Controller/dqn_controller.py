import random
from typing import List, Type, Optional

import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


GAMMA = 0.9
LEARNING_RATE = 0.01

MEMORY_SIZE = 1000000
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.9999
NUMBER_OF_WORKERS = 8


class DQNController:

    def __init__(self,
                 nn_model: Sequential,
                 action_space_size: int,
                 observation_space_size: int,
                 callbacks: Optional[List] = None
                 ):

        self.model = nn_model
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size

        self.exploration_rate = EXPLORATION_MAX

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.callbacks_list = callbacks
        self.parameter_history = []

    @classmethod
    def load(cls,
             model_file: str,
             action_space_size: int,
             observation_space_size: int,
             callbacks: Optional[List] = None):

        model = tf.keras.models.load_model(model_file)

        return cls(nn_model=model,
                   action_space_size=action_space_size,
                   observation_space_size=observation_space_size,
                   callbacks=callbacks)

    @classmethod
    def new_model(cls,
                  observation_space_size: int,
                  action_space_size: int,
                  checkpoint_name: str,):

        nn_model = Sequential()
        nn_model.add(Dense(24, input_shape=(observation_space_size,), activation="tanh"))
        nn_model.add(Dense(24, activation="tanh"))
        nn_model.add(Dense(24, activation="tanh"))
        nn_model.add(Dense(action_space_size, activation="linear"))
        nn_model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f"./Checkpoints/{checkpoint_name}.ckpt",
                                                        verbose=0,
                                                        save_freq=10000)
        callbacks_list = [checkpoint]

        return cls(nn_model=nn_model,
                   action_space_size=action_space_size,
                   observation_space_size=observation_space_size,
                   callbacks=callbacks_list
                   )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act_or_explore(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def act(self, state):
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
            self.model.fit(x=state,
                           y=q_values,
                           verbose=0,
                           callbacks=self.callbacks_list,
                           use_multiprocessing=True,
                           workers=NUMBER_OF_WORKERS)
        print("Replay complete!")
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def load_from_checkpoint(self, checkpoint_file: str):
        try:
            self.model.load_weights(checkpoint_file)
        except FileNotFoundError:
            print("Incorrect file specified!")
