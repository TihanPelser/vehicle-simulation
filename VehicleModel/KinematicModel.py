import numpy as np
import math
import pandas as pd


class KinematicVehicleModel:

    def __init__(self, dt):
        # PARAMETERS
        self.max_steering_angle = 25  # Max steering angle in degrees
        # Mass, Inertia
        # Geometry
        self.lf = 40.109 / 100
        self.lr = 34.891 / 100
        self.df = 67.8 / 100
        self.dr = 18 / 100

        self.global_x = 0
        self.global_y = 0
        self.global_theta = 0
        self.delta = 0

        self.velocity = 2.78
        self.acceleration = 0

        self.dt = dt
        self.timestamp = 0

        self.data = []

    def reset(self):

        self.global_x = 0
        self.global_y = 0
        self.global_theta = 0
        self.delta = 0

        self.velocity = 2.78
        self.acceleration = 0

        self.max_steering_angle = 25

        self.timestamp = 0

        self.data = []

    def set_position(self, x, y, heading):
        self.global_x = x
        self.global_y = y
        self.global_theta = heading

    def _set_steering_angle(self, delta):
        if delta > np.deg2rad(self.max_steering_angle):
            self.delta = np.deg2rad(25)
        elif delta < - np.deg2rad(self.max_steering_angle):
            self.delta = - np.deg2rad(25)
        else:
            self.delta = delta

    def _calculate_all(self):
        # compute slip angle
        beta = np.arctan(self.lf / (self.lf + self.lr) * np.tan(self.delta))

        # compute next state
        delta_x = self.dt * (self.velocity * np.cos(self.global_theta + beta))
        delta_y = self.dt * (self.velocity * np.sin(self.global_theta + beta))
        delta_theta = self.dt * (self.velocity / self.lr) * np.sin(beta)
        delta_v = self.dt * self.acceleration

        self.global_x += delta_x
        self.global_y += delta_y
        self.global_theta += delta_theta
        self.velocity += delta_v

        current_iteration_data = [self.timestamp, self.global_x, self.global_y, self.global_theta, self.delta,
                                  beta, delta_theta, 0, 0, 0, 0]

        self.data.append(current_iteration_data)

    # MAIN DRIVE FUNCTION
    def drive(self, steering_angle, ):
        self._set_steering_angle(steering_angle)
        self._calculate_all()
        self.timestamp += self.dt
        return self.get_status()

    def get_status(self):
        return np.array([self.global_x, self.global_y, self.global_theta])

    def parameter_history(self) -> pd.DataFrame:
        data = pd.DataFrame(self.data, columns=["Time", "x", "y", "Theta", "Delta", "Beta", "Yaw Rate",
                                                "Slip Front", "Slip Rear", "Force Front", "Force Rear"])
        return data