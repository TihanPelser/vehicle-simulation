import numpy as np
import math
import pandas as pd
from TyreModel.LinearCutoff import LinearTyre

class DynamicVehicleModel:
    def __init__(self, dt: float, tyre_model: LinearTyre):
        # PARAMETERS
        self.max_steering_angle = 25  # Max steering angle in degrees
        # Mass, Inertia
        # Mass, Inertia and Velocity
        self.m = 124.5
        self.Izz = 44
        self.V = 2.78  # Speed [m/s]

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
