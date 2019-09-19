from VehicleModel.Vehicle import Vehicle
from Controller.DQNController import DQNController
from matplotlib import style
import pandas as pd
import numpy as np
import logging
from typing import Optional, List
from collections import namedtuple
import time


class Simulation:
    # logging.basicConfig(format='%(levelname)s-%(message)s')

    # CUSTOM DATA TYPES
    #  NamedTuple to save initial setup state for reset
    Initial = namedtuple("Initial", "name input_type input_data dt timeout debug waypoint_threshold")
    #  NamedTuple to save closest points and index of closest point in all points
    ClosestPoints = namedtuple("ClosestPoints", "p1 p2 start_index")

    controller: DQNController
    vehicle: Vehicle

    # Simulation parameters
    debug: bool
    dt: float  # Time step [s]
    timeout: float
    input_type: str
    input_data: np.ndarray
    waypoint_threshold: float
    episode: int
    simulation_name: str
    run_time: float
    results: Optional[pd.DataFrame]

    closest_points: ClosestPoints
    number_of_points: int
    prev_distance: Optional[float]
    prev_angle: Optional[float]
    iterations_without_progress: int
    terminal: bool

    style.use('fast')

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)

########################################################################################################################
# ----------------------------------------------SETUP FUNCTIONS------------------------------------------------------- #
########################################################################################################################

    def __init__(self, sim_name: str, vehicle: Vehicle,  input_type: str, input_data: np.ndarray, timestep: float,
                 timeout: float, waypoint_threshold: float = 0.5, debug: bool = True):
        try:
            self.vehicle = vehicle
            self.initial = self.Initial(name=sim_name, input_type=input_type, input_data=input_data, dt=timestep,
                                        timeout=timeout, debug=debug, waypoint_threshold=waypoint_threshold)
            self.episode = 0
            self.points_reached = 0
            self.has_been_reset = False

            # self.reset()

        except KeyError as e:
            logging.error("Config file missing/incorrect")
            print("Key Error")
            logging.error(e)
            exit(1)

    def reset(self, save_data: bool = False):
        self.has_been_reset = True
        self.episode += 1
        self.points_reached = 0

        self.dt = self.initial.dt
        self.timeout = self.initial.timeout
        self.input_type = self.initial.input_type
        self.input_data = self.initial.input_data
        self.debug = self.initial.debug
        self.waypoint_threshold = self.initial.waypoint_threshold
        self.run_time = 0.
        self.results: Optional[pd.DataFrame] = None

        name = self.initial.name
        t = time.strftime("%H-%M-%S", time.gmtime())
        base_name = f"{name}_{t}"
        self.simulation_name = f"{base_name}_{self.episode}"

        self.closest_points = self.ClosestPoints(p1=self.input_data[0], p2= self.input_data[1], start_index=0)
        self.number_of_points = len(self.input_data)

        self.prev_distance = None
        self.prev_angle = None
        self.iterations_without_progress = 0
        self.terminal = False

        if save_data:
            self.vehicle.reset(save_name=f"{self.simulation_name}")
        else:
            self.vehicle.reset()

        vehicle_status = self.vehicle.get_status()
        vehicle_coords = vehicle_status[0:2]
        vehicle_heading = vehicle_status[2]
        state = self._get_state(vehicle_coords=vehicle_coords, vehicle_heading=vehicle_heading)

        return state

########################################################################################################################
# -------------------------------------------CALCULATION FUNCTIONS---------------------------------------------------- #
########################################################################################################################

    def _update_closest_points(self, vehicle_pos) -> bool:
        """
        Checks whether the vehicle has arrived at the next point and updates the 'closest_points' list accordingly
        :param vehicle_pos: Global position of the vehicle
        :return: Bool stating whether the next closest point is the final point
        """
        p1 = self.closest_points.p1
        d1 = self._calculate_distances(vehicle_pos=vehicle_pos, p1=p1, is_final=True)[0]

        if d1 <= self.waypoint_threshold:
            ind = self.closest_points.start_index + 1
            self.points_reached += 1
            if ind == self.number_of_points:
                new_p1 = self.input_data[ind]
                self.closest_points = self.ClosestPoints(p1=new_p1, p2=None, start_index=ind)
                return True
            else:
                new_p1 = self.input_data[ind]
                new_p2 = self.input_data[ind + 1]
                self.closest_points = self.ClosestPoints(p1=new_p1, p2=new_p1, start_index=ind)
                return False

    def _calculate_angles(self, vehicle_pos: np.ndarray, p1: np.ndarray, p2: Optional[np.ndarray] = None,
                          is_final: bool = False) -> float(2):
        """
        Calculates the angles between the current vehicle position to the next 2 points. If only one point remains to
        be reached, the argument 'is_final' should be set to True. Only a single angle will then be returned, with the
        second angle returned as 0
        :param vehicle_pos: Coords of vehicle
        :param p1: Coords of closest point
        :param p2: Coords of second closest point
        :return: Global reference angles (in radians) between vehicle and points
        """
        delta_x1 = p1[0] - vehicle_pos[0]
        delta_y1 = p1[1] - vehicle_pos[1]
        theta1 = np.arctan2(delta_y1, delta_x1)
        theta2 = 0
        if not is_final:
            delta_x2 = p2[0] - vehicle_pos[0]
            delta_y2 = p2[1] - vehicle_pos[1]
            theta2 = np.arctan2(delta_y2, delta_x2)

        return theta1, theta2

    def _calculate_distances(self, vehicle_pos: np.ndarray, p1: np.ndarray, p2: Optional[np.ndarray] = None,
                             is_final: bool = False) -> float(2):
        """
        Calculates the distances between the current vehicle position to the next 2 points. If only one point remains to
        be reached, the argument 'is_final' should be set to True. Only a single distance will then be returned,
        with the second distance returned as 0
        :param vehicle_pos: Coords of vehicle
        :param p1: Coords of closest point
        :param p2: Coords of second closest point
        :return: Distance from vehicle and points
        """
        delta_x1 = p1[0] - vehicle_pos[0]
        delta_y1 = p1[1] - vehicle_pos[1]
        d1 = np.sqrt(delta_x1**2 + delta_y1**2)
        d2 = 0
        if not is_final:
            delta_x2 = p2[0] - vehicle_pos[0]
            delta_y2 = p2[1] - vehicle_pos[1]
            d2 = np.sqrt(delta_x2**2 + delta_y2**2)

        return d1, d2

    def _calculate_heading_errors(self, vehicle_heading: float, theta1: float, theta2: float,
                                  is_final: bool = False) -> float(2):
        """
        Calculates the differences (errors) between the vehicle heading and the headings (global referenced angles)
        to the points. If only one point remains to be reached, the parameter 'is_final' should be set to 0. Only one
        error value will then be returned, with the second error value set to 0.
        :param vehicle_heading: Global heading of the vehicle (radians)
        :param theta1: Global referenced angle between the vehicle and the closest point (radians)
        :param theta2: Global referenced angle between the vehicle and the second closest point (radians)
        :param is_final: Boolean value stating whether the next point is the final point
        :return: Heading errors in radians
        """
        theta1_error = vehicle_heading - theta1
        theta2_error = 0.
        if not is_final:
            theta2_error = vehicle_heading - theta2

        return theta1_error, theta2_error

    def _get_state(self, vehicle_coords: np.ndarray, vehicle_heading: float):

        is_final = self._update_closest_points(vehicle_pos=vehicle_coords)
        p1 = self.closest_points.p1
        p2 = self.closest_points.p2

        d1, d2 = self._calculate_distances(vehicle_pos=vehicle_coords, p1=p1, p2=p2, is_final=is_final)
        theta1, theta2 = self._calculate_angles(vehicle_pos=vehicle_coords, p1=p1, p2=p2, is_final=is_final)
        theta1_error, theta2_error = self._calculate_heading_errors(vehicle_heading=vehicle_heading, theta1=theta1,
                                                                    theta2=theta2, is_final=is_final)

        return np.array([d1, d2, theta1_error, theta2_error])

    def _calculate_reward(self, reward_type: str, state: np.ndarray) -> int:
        # Previous distance reward system

        d1 = state[0]
        d2 = state[1]
        theta1 = state[2]
        theta2 = state[3]

        if reward_type == "distance":
            if self.prev_distance is not None:
                if d1 >= self.prev_distance:
                    self.iterations_without_progress += 1
                    reward = -1
                else:
                    self.prev_distance = d1
                    self.iterations_without_progress = 0
                    reward = 1
            else:
                self.prev_distance = d1
                self.iterations_without_progress = 0
                reward = 1
            return reward

        elif reward_type == "angle":
            # Angle on target reward system

            if self.prev_angle is not None:
                if theta1 >= self.prev_angle:
                    self.iterations_without_progress += 1
                    self.prev_angle = theta1
                    reward = 0
                else:
                    self.prev_angle = theta1
                    self.iterations_without_progress = 0
                    reward = 1
            else:
                self.prev_angle = theta1
                self.iterations_without_progress = 0
                reward = 1

            return reward
########################################################################################################################
# -------------------------------------------------RUN FUNCTIONS------------------------------------------------------ #
########################################################################################################################

    def step(self, step_type: str, input: Optional = None, speed: float = None):

        if self.terminal:
            logging.error("Terminal state has been reached. Please call sim.reset() in order to restart the simulation")
            return None
        if not self.has_been_reset:
            logging.error("Please call .reset() before calling .step()")
            return None

        if step_type == "action":
            # Actions are encoded as 0 = Left and 1 = Right
            if input == 0:
                angle = self.vehicle.delta - np.radians(1)
            elif input == 1:
                angle = self.vehicle.delta + np.radians(1)
            else:
                logging.error("Invalid action")
                return None
        elif step_type == "steer":
            angle = input
        else:
            logging.error("Parameters must contain either steering_angle or action")
            return None

        vehicle_status = self.vehicle.drive(angle, speed)
        vehicle_coords = vehicle_status[0:2]
        vehicle_heading = vehicle_status[2]

        state = self._get_state(vehicle_coords=vehicle_coords, vehicle_heading=vehicle_heading)

        self.run_time += self.dt

        reward = self._calculate_reward(reward_type="angle", state=state)

        if self.iterations_without_progress >= 100:
            self.terminal = True

        if self.run_time >= self.timeout:
            self.terminal = True

        return state, reward, self.points_reached, self.terminal, self.run_time

    def _run_steering(self) -> pd.DataFrame:
        for angle in self.input_data:
            self.vehicle.drive(steering_angle=angle)
            self.run_time += self.dt

        return self.vehicle.parameter_history()

########################################################################################################################
# ---------------------------------------------LOGGING FUNCTIONS------------------------------------------------------ #
########################################################################################################################

    def log_error_information(self):
        slip_angle_exceeded = self.vehicle.tyre.exceeded_slip_angle_max_count
        steering_angle_exceeded = self.vehicle.exceeded_steering_angle_max_count

        if self.debug:
            with open(f"Results/debug/{self.simulation_name}.txt", "w+") as log:
                log.write("During simulation the following errors/warnings occurred:\n")
                log.write(f"\tThe calculated slip angle exceed the maximum values tested experimentally "
                          f"{slip_angle_exceeded} times\n")
                log.write(f"\tThe maximum steering angle input exceeded the maximum allowable angle "
                          f"{steering_angle_exceeded} times")
        else:
            logging.warning("During simulation the following errors/warnings occurred:")
            logging.warning(f"\tThe calculated slip angle exceed the maximum values tested experimentally "
                            f"{slip_angle_exceeded} times")
            logging.warning(f"\tThe maximum steering angle input exceeded the maximum allowable angle "
                            f"{steering_angle_exceeded} times")
