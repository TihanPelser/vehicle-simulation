import numpy as np
from TyreModels.LinearCutoffTyreModel import LinearTyre
from typing import Optional, List
from collections import namedtuple
import logging
import pandas as pd
import time


class Vehicle:

    # Initial setup values object (NamedTuple)
    Initial = namedtuple("Initial", "x y dt theta speed")

    # PARAMETERS

    # Mass, Inertia and Velocity
    m: float
    Izz: float
    V: float  # Speed [m/s]

    # Geometry
    lf: float
    lr: float
    df: float
    dr: float

    #############################
    # Updated at each time step #
    #############################

    # Dynamics

    delta: float  # Steering angle [rad]
    beta: float  # Velocity Angle (Heading) [rad]
    slip_front: float  # Slip angle front [rad]
    slip_rear: float  # Slip angle rear [rad]

    beta_fl: float  # Slip angle front left [rad]
    beta_fr: float  # Slip angle front right [rad]
    beta_rl: float  # Slip angle rear left [rad]
    beta_rr: float  # Slip angle rear right [rad]

    y_double_dot: float  # Lateral acceleration [m/s^2]
    y_dot: float  # Lateral velocity [m/s]
    y_delta: float  # Lateral position delta [m]
    theta: float  # Yaw angle [rad]
    r: float  # Yaw rate [rad/s]
    r_dot: float  # Yaw acceleration [rad/s^2]

    force_front: float  # Force experienced at front tyres [N]
    force_rear: float  # Force experienced at rear tyres [N]

    # Logging and Statistics
    data_list = []  # List of all vehicle data per time step
    logging_time: float  # Time taken to log all data [s]
    run_time: float  # Time taken to run [s]
    force_calc_time: float  # Time taken to calculate forces [s]
    timestamp: float  # Current timestamp [s]

    tyre: LinearTyre  # Tyre model to use
    dt: float  # Time step size [s]
    global_x: float  # Global x - coord [m]
    global_y: float  # Global y - coord [m]
    global_theta: float  # Global heading [rad]
    exceeded_steering_angle_max_count: int

    def __init__(self, tyre_model: LinearTyre, dt: float, x0: float = 0., y0: float = 0.,
                 theta0: float = 0., speed: float = 2.78):

        #  TODO: Rework __init__() to use reset() as both essentially do the same
        try:
            # Save initial creation values to reset vehicle if needed
            self.tyre = tyre_model
            self.max_steering_angle = 25  # Max steering angle in degrees

            self.initial = self.Initial(x=x0, y=y0, theta=theta0, dt=dt, speed=speed)
            self.reset()

        except KeyError as e:
            logging.error("Vehicle configuration file incorrect/missing.")
            logging.error(e)
            exit(1)

    def reset(self, save_name: Optional[str] = None):

        if save_name is not None and len(self.data_list) > 0:
            params = self.parameter_history()
            params.to_csv(f"SavedData/vehicledata_{save_name}.csv")

        self.tyre.reset(save_name=save_name)

        # PARAMETERS

        # Mass, Inertia and Velocity
        self.m = 124.5
        self.Izz = 44
        self.V = self.initial.speed  # Speed [m/s]

        # Geometry
        self.lf = 40.109 / 100
        self.lr = 34.891 / 100
        self.df = 67.8 / 100
        self.dr = 18 / 100

        #############################
        # Updated at each time step #
        #############################

        # Dynamics

        self.delta = 0  # Steering angle [rad]
        self.beta = 0  # Velocity Angle (Heading) [rad]
        self.slip_front = 0.  # Slip angle front [rad]
        self.slip_rear = 0.  # Slip angle rear [rad]

        self.beta_fl = 0.  # Slip angle front left [rad]
        self.beta_fr = 0.  # Slip angle front right [rad]
        self.beta_rl = 0.  # Slip angle rear left [rad]
        self.beta_rr = 0.  # Slip angle rear right [rad]

        self.y_double_dot = 0  # Lateral acceleration [m/s^2]
        self.y_dot = 0  # Lateral velocity [m/s]
        self.y_delta = 0  # Lateral position delta [m]
        self.theta = 0  # Yaw angle [rad]
        self.r = 0  # Yaw rate [rad/s]
        self.r_dot = 0  # Yaw acceleration [rad/s^2]

        self.force_front = 0.  # Force experienced at front tyres [N]
        self.force_rear = 0.  # Force experienced at rear tyres [N]

        # Logging and Statistics
        self.data_list = []  # List of all vehicle data per time step
        self.logging_time = 0.  # Time taken to log all data [s]
        self.run_time = 0.  # Time taken to run [s]
        self.force_calc_time = 0.  # Time taken to calculate forces [s]
        self.timestamp = 0.  # Current timestamp [s]

        self.dt = self.initial.dt  # Time step size [s]
        self.global_x = self.initial.x  # Global x - coord [m]
        self.global_y = self.initial.y  # Global y - coord [m]
        self.theta = self.initial.theta  # Heading delta of vehicle for current time step [rad]
        self.global_theta = self.initial.theta  # Global heading [rad]
        self.exceeded_steering_angle_max_count = 0
        self.data_list.append([self.timestamp, self.global_x, self.global_y, self.global_theta, self.delta,
                               self.beta, self.r, self.slip_front, self.slip_rear, self.force_front,
                               self.force_rear])

########################################################################################################################
    # Integrals of Basic Equations of Motion

    def _set_lateral_acceleration(self, forces: np.ndarray):
        """

        :param forces: array containing the forces F_fl, F_fr, F_rl, F_rr
        :param dt: timestep
        :return:
        """
        self.y_double_dot = sum(forces) / self.m

    def _set_lateral_velocity(self):
        self.y_dot = self.y_double_dot * self.dt

    def _set_lateral_position_delta(self):
        self.y_delta = self.y_dot * self.dt

    def _set_yaw_acceleration(self, forces: np.ndarray):
        """

        :param forces:
        :return:
        """
        self.r_dot = ((2 * self.lf * forces[0] - 2 * self.lr * forces[1]) / self.Izz)

    def _set_yaw_rate(self):
        self.r = self.r_dot * self.dt

    def _set_yaw_angle(self):
        self.theta = self.r * self.dt

    # DEPRECATED

    # def front_axle_slip_angle(self) -> None:
    #     self.slip_front = self.beta + (self.lf * self.r/self.V) - self.delta
    #
    # def rear_axle_slip_angle(self) -> None:
    #     self.slip_rear = self.beta + (self.lr * self.r/self.V)

    def _slip_angles(self) -> np.ndarray:
        # self.beta_fl = (self.V * self.beta + self.lf * self.r)/(self.V - self.df * self.r/2) - self.delta
        # self.beta_fr = (self.V * self.beta + self.lf * self.r) / (self.V + self.df * self.r / 2) - self.delta
        #
        # self.beta_rl = (self.V * self.beta - self.lf * self.r) / (self.V - self.df * self.r / 2)
        # self.beta_rr = (self.V * self.beta - self.lf * self.r) / (self.V + self.df * self.r / 2)
        # return np.array([self.beta_fl, self.beta_fr, self.beta_rl, self.beta_rr])

        self.slip_front = self.beta + (self.lf * self.r / self.V) - self.delta
        self.slip_rear = self.beta + (self.lr * self.r / self.V)

        return np.array([self.slip_front, self.slip_rear])

    def _normal_forces(self) -> np.ndarray:
        """
        Only used when load transfer is considered
        :return:
        """
        return np.array([26.5 + 28.5, 35 + 35]) * 9.81

    def _set_steering_angle(self, delta):
        """
        Set the steering angle of the vehicle. Due to the geometry of the platform, there is a hard limit to the
        steering of 30 degrees. If an input greater than the allowed limit is applied, an error is logged and the
        steering set to the limit of 30 degrees.
        :param delta: Steering input [rad]
        :return: None
        """
        if np.degrees(delta) > self.max_steering_angle:
            # logging.warning("Steering input exceeds steering limit. Applied steering will be limited.")
            self.exceeded_steering_angle_max_count += 1
            self.delta = np.deg2rad(self.max_steering_angle)

        elif np.degrees(delta) < - self.max_steering_angle:
            # logging.warning("Steering input exceeds steering limit. Applied steering will be limited.")
            self.exceeded_steering_angle_max_count += 1
            self.delta = - np.deg2rad(self.max_steering_angle)
        else:
            self.delta = delta

    def _update_beta(self):
        # beta_dot = self.y_doubledot/self.V - self.r
        #
        # self.beta = beta_dot * self.dt
        self.beta = np.arcsin(self.y_dot/self.V)

    def _calculate_all(self, steering_angle: float):

        start = time.time()

        # Set steering angle
        self._set_steering_angle(steering_angle)

        # Update beta for next run
        self._update_beta()

        # Calculate slip angles DEPRECATED
        # self.front_axle_slip_angle()
        # self.rear_axle_slip_angle()

        # Calculate slip angles and Normal Forces
        slip = self._slip_angles()
        normal = self._normal_forces()

        # Calculate lateral forces
        t_force_start = time.time()
        forces = self.tyre.calculate_lateral_force(slip_angles=slip, normal=normal, time=self.timestamp)
        self.force_front = forces[0]
        self.force_rear = forces[1]

        self.force_calc_time += (time.time() - t_force_start)

        # Calculate acceleration -> Velocity -> Position Delta
        self._set_lateral_acceleration(forces=forces)
        self._set_lateral_velocity()
        self._set_lateral_position_delta()
        self._set_yaw_acceleration(forces=forces)
        self._set_yaw_rate()
        self._set_yaw_angle()

        end_run = time.time()

        end_logging = time.time()

        self.logging_time += end_logging-end_run
        self.run_time += end_run - start

    def drive(self, steering_angle: float):

        self._calculate_all(steering_angle=steering_angle)

        # Global heading change
        self.global_theta += self.theta

        # Longitudinal Velocity
        delta_x_v = self.V * np.cos(self.beta) * self.dt * np.cos(self.global_theta)
        delta_y_v = self.V * np.cos(self.beta) * self.dt * np.sin(self.global_theta)

        # Global position changes

        self.global_x += delta_x_v + self.y_delta * np.cos(self.global_theta + np.pi/2)
        self.global_y += delta_y_v + self.y_delta * np.sin(self.global_theta + np.pi/2)

        self.timestamp += self.dt

        current_iteration_data = [self.timestamp, self.global_x, self.global_y, self.global_theta, self.delta,
                                  self.beta, self.r, self.slip_front, self.slip_rear, self.force_front, self.force_rear]
        # print(current_iteration_data)
        if np.isnan(current_iteration_data).any():
            logging.error("NaN value detected in current iteration.")

        self.data_list.append(current_iteration_data)

        return self.get_status()

    def parameter_history(self) -> pd.DataFrame:
        data = pd.DataFrame(self.data_list, columns=["Time", "x", "y", "Theta", "Delta", "Beta", "Yaw Rate",
                                                     "Slip Front", "Slip Rear", "Force Front", "Force Rear"])
        return data

    def get_status(self):
        return np.array([self.global_x, self.global_y, self.global_theta])
