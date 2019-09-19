import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from keras.models import Sequential
from keras.layers import Dense, Activation


class NNTyreModel:
    """
    NN Approximated Tyre Model which provides functions to calculate the generated lateral force, given a slip angle
    and normal load
    """

    xy: np.ndarray
    z: np.array
    model = None

    def __init__(self, config: dict):
        try:
            # data = config["data"]
            self.data = pd.read_csv(f"./TyreModelNN/Tyre_Model.txt", sep="\t").values
            self.xy = np.stack((np.radians(self.data[:, 0]), self.data[:, 2]), axis=1)
            # print(self.xy)
            self.z = self.data[:, 1]
            self.exceeded_slip_angle_max_count = 0
            # self.data = []
            self.cutoff = False
            # Keras Model WIP
            # self.model = Sequential([
            #     Dense(32, input_shape=(2,)),
            #     Activation('relu'),
            #     Dense(10),
            #     Activation('softmax'),
            # ])
            # self.model.compile(loss='mean_squared_error',
            #               optimizer='sgd',
            #               metrics=['mae', 'acc'])

            self.model = MLPRegressor(hidden_layer_sizes=(1000, 1000, 1000, 1000), solver='lbfgs', max_iter=10000, alpha=1E-5)
            X_train, X_test, y_train, y_test = train_test_split(self.xy, self.z, test_size=0.01, random_state=42)

            self.model.fit(self.xy, self.z)

        except FileNotFoundError:
            logging.error("Tyre model data file not found. Check path in config file")
            exit(1)
        except KeyError as e:
            logging.error("Tyre config file missing/incorrect")
            logging.error(e)
            exit(1)
