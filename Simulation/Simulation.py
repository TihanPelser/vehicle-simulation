from VehicleModel.KinematicModel import KinematicVehicleModel
from VehicleModel.DynamicModel import DynamicVehicleModel
from Controller.DQNController import DQNController
import pandas as pd
import numpy as np
import logging
from typing import Optional, List
from collections import namedtuple
import time
import pygame
import sys
import random


class Simulation:

    # CUSTOM DATA TYPES
    #  NamedTuple to save initial setup state for reset
    Initial = namedtuple("Initial", "name input_data dt timeout iterations_per_step "
                                    "way_point_threshold distance_between_points")
    StepReturn = namedtuple("StepReturn", "state reward progress terminal run_time end_condition")
    #  NamedTuple to save closest points and index of closest point in all points
    ClosestPoints = namedtuple("ClosestPoints", "p1 p2 start_index")

    _vehicle_coords: List = []

    # Simulation parameters
    debug: bool
    dt: float  # Time step [s]
    timeout: float
    input_type: str
    input_data: np.ndarray
    way_point_threshold: float
    episode: int
    simulation_name: str
    run_time: float
    results: List

    closest_points: ClosestPoints
    number_of_points: int
    prev_distance: Optional[float]
    prev_angle: Optional[float]
    iterations_without_progress: int
    terminal: bool
    solved: bool
    is_final: bool
    way_point_reached_in_step: bool
    current_run: int
    iterations_per_step: int
    distance_between_points: float
    last_state: np.ndarray

    # PyGame Settings
    SIZE = 1280, 720
    _screen: Optional[pygame.Surface]
    _vehicle_sprite: Optional[pygame.Surface]
    _vehicle_sprite_rect: Optional[pygame.Rect]
    _vehicle_sprite_angle: Optional[float]
    _path_origin: Optional[tuple]
    _scaling_factor: Optional[float]
    _scaling_factor: Optional[float]
    _data_font: Optional[pygame.font.Font]
    _heading_font: Optional[pygame.font.Font]
    _text_box: Optional[pygame.Rect]
    _way_point_render_radius: Optional[int]

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)

########################################################################################################################
# ----------------------------------------------SETUP FUNCTIONS------------------------------------------------------- #
########################################################################################################################

    def __init__(self, sim_name: str, vehicle: Optional[KinematicVehicleModel],
                 input_data: np.ndarray, time_step: float, timeout: float, iterations_per_step: int,
                 distance_between_points: float, way_point_threshold: float = 0.5):
        self.vehicle = vehicle
        self.initial = self.Initial(name=sim_name, input_data=input_data, dt=time_step,
                                    timeout=timeout, iterations_per_step=iterations_per_step,
                                    way_point_threshold=way_point_threshold,
                                    distance_between_points=distance_between_points)
        self.episode = 0
        self.points_reached = 0
        self.has_been_reset = False
        self.current_run = 0
        pygame.init()

        print("Sim created...")
        print(f"Number of Waypoints: {len(input_data)}")

    def reset(self):

        self.has_been_reset = True
        self.episode += 1
        self.points_reached = 0
        self._vehicle_coords = []

        self.vehicle.reset()

        self.dt = self.initial.dt
        self.timeout = self.initial.timeout
        self.input_data = self.initial.input_data
        self.iterations_per_step = self.initial.iterations_per_step
        self.way_point_threshold = self.initial.way_point_threshold
        self.distance_between_points = self.initial.distance_between_points
        self.run_time = 0.
        self.results = []

        # Sim name
        name = self.initial.name
        t = time.strftime("%H-%M-%S", time.gmtime())
        base_name = f"{name}_{t}"
        self.simulation_name = f"{base_name}_{self.episode}"

        self.closest_points = self.ClosestPoints(p1=self.input_data[1], p2=self.input_data[2], start_index=1)
        self.number_of_points = len(self.input_data) - 1
        self._set_vehicle_position(random_state=True)

        self.prev_distance = None
        self.prev_angle = None
        self.iterations_without_progress = 0
        self.terminal = False
        self.solved = False
        self.is_final = False
        self.way_point_reached_in_step = False
        self.current_run += 1

        vehicle_status = self.vehicle.get_status()
        vehicle_coords = vehicle_status[0:2]
        vehicle_heading = vehicle_status[2]
        self._get_state(vehicle_coords=vehicle_coords, vehicle_heading=vehicle_heading)

        # Rendering
        self._screen = None
        self._vehicle_sprite = None
        self._vehicle_sprite_rect = None
        self._path_origin = None
        self._heading_font = None
        self._data_font = None
        self._text_box = None
        self._vehicle_sprite_angle = None
        self._way_point_render_radius = None

        return self.last_state

    def _set_vehicle_position(self, random_state: bool = False):
        x = self.input_data[0][0]
        y = self.input_data[0][1]
        delta_x = self.closest_points.p1[0] - x
        delta_y = self.closest_points.p1[1] - y

        heading = np.arctan2(delta_y, delta_x)
        if random_state:
            # Add random noise between -65 and 65 to initial heading
            noise = np.deg2rad((random.random() * 130) - 65)
            heading += noise

        self.vehicle.set_position(x, y, heading)

########################################################################################################################
# -------------------------------------------CALCULATION FUNCTIONS---------------------------------------------------- #
########################################################################################################################

    def _update_closest_points(self, vehicle_pos):
        """
        Checks whether the vehicle has arrived at the next point and updates the 'closest_points' list accordingly
        :param vehicle_pos: Global position of the vehicle
        :return: Bool stating whether the next closest point is the final point
        """
        self.way_point_reached_in_step = False
        p1 = self.closest_points.p1
        d1, _ = self._calculate_distances(vehicle_pos=vehicle_pos, p1=p1, single_calc=True)

        if d1 <= self.way_point_threshold:
            self.way_point_reached_in_step = True
            ind = self.closest_points.start_index + 1
            self.points_reached += 1
            if self.is_final:
                # If is_final then only one point remains
                # Set to solved if final point reached
                self.solved = True
                self.is_final = True
            elif ind == self.number_of_points - 1:
                new_p1 = self.input_data[ind]
                self.closest_points = self.ClosestPoints(p1=new_p1, p2=None, start_index=ind)
                self.is_final = True
            else:
                new_p1 = self.input_data[ind]
                new_p2 = self.input_data[ind + 1]
                self.closest_points = self.ClosestPoints(p1=new_p1, p2=new_p2, start_index=ind)
                self.is_final = False

    def _calculate_angles(self, vehicle_pos: np.ndarray, p1: np.ndarray, p2: Optional[np.ndarray] = None) -> float(2):
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
        if not self.is_final:
            delta_x2 = p2[0] - vehicle_pos[0]
            delta_y2 = p2[1] - vehicle_pos[1]
            theta2 = np.arctan2(delta_y2, delta_x2)

        return theta1, theta2

    def _calculate_distances(self, vehicle_pos: np.ndarray, p1: np.ndarray,
                             p2: Optional[np.ndarray] = None, single_calc: bool = False) -> float(2):
        """
        Calculates the distances between the current vehicle position to the next 2 points.
        :param vehicle_pos: Coords of vehicle
        :param p1: Coords of closest point
        :param p2: Coords of second closest point
        :return: Distance from vehicle and points
        """
        # print(f"Single Calc: {single_calc}")
        # print(f"Is Final: {self.is_final}")
        delta_x1 = p1[0] - vehicle_pos[0]
        delta_y1 = p1[1] - vehicle_pos[1]
        d1 = np.sqrt(delta_x1 ** 2 + delta_y1 ** 2)
        d2 = 0
        if single_calc:
            return d1, d2
        if not self.is_final:
            # print(f"Entering second point calc")
            delta_x2 = p2[0] - vehicle_pos[0]
            delta_y2 = p2[1] - vehicle_pos[1]
            d2 = np.sqrt(delta_x2 ** 2 + delta_y2 ** 2)

        return d1, d2

    def _calculate_heading_errors(self, vehicle_heading: float, theta1: float, theta2: float) -> float(2):
        """
        Calculates the differences (errors) between the vehicle heading and the headings (global referenced angles)
        to the points. If only one point remains to be reached, the parameter 'is_final' should be set to 0. Only one
        error value will then be returned, with the second error value set to 0.
        :param vehicle_heading: Global heading of the vehicle (radians)
        :param theta1: Global referenced angle between the vehicle and the closest point (radians)
        :param theta2: Global referenced angle between the vehicle and the second closest point (radians)
        :return: Heading errors in radians
        """
        theta1_error = theta1 - vehicle_heading
        theta2_error = 0.
        if not self.is_final:
            theta2_error = theta2 - vehicle_heading

        return theta1_error, theta2_error

    def _get_state(self, vehicle_coords: np.ndarray, vehicle_heading: float):

        self._update_closest_points(vehicle_pos=vehicle_coords)

        if self.solved:
            self.last_state = np.array([0., 0., 0., 0.])

        p1 = self.closest_points.p1
        p2 = self.closest_points.p2

        d1, d2 = self._calculate_distances(vehicle_pos=vehicle_coords, p1=p1, p2=p2)
        theta1, theta2 = self._calculate_angles(vehicle_pos=vehicle_coords, p1=p1, p2=p2)
        theta1_error, theta2_error = self._calculate_heading_errors(vehicle_heading=vehicle_heading, theta1=theta1,
                                                                    theta2=theta2)
        # Normalise distance
        normalised_d1 = d1 / self.distance_between_points
        normalised_d2 = d2 / self.distance_between_points * 2

        # Inverse Method
        # inverse_angle_1 = np.sign(theta1_error) * (np.deg2rad(120) - abs(theta1_error))
        # inverse_angle_2 = np.sign(theta2_error) * (np.deg2rad(120) - abs(theta2_error))

        inverse_d1 = 1 - normalised_d1
        inverse_d2 = 1 - normalised_d2

        # self.last_state = np.array([normalised_d1, normalised_d2, theta1_error, theta2_error])

        # State consists of inverse, normalised distances and heading errors
        self.last_state = np.array([inverse_d1, inverse_d2, np.rad2deg(theta1_error), np.rad2deg(theta2_error),
                                    np.rad2deg(vehicle_heading)])

    def _calculate_reward(self, reward_type: str, ) -> int:

        reward = 0

        d1 = self.last_state[0]
        d2 = self.last_state[1]
        theta1 = self.last_state[2]
        theta2 = self.last_state[3]

        # Previous distance reward system
        if reward_type == "distance":
            if self.prev_distance is not None:
                if d1 >= self.prev_distance:
                    self.iterations_without_progress += 1
                    self.prev_distance = d1
                    reward = -1
                else:
                    self.prev_distance = d1
                    self.iterations_without_progress = 0
                    reward = 1
            else:
                self.prev_distance = d1
                self.iterations_without_progress = 0
                reward = 1

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

        elif reward_type == "granular-angle":
            reward = 5 - abs(np.degrees(theta1))

        elif reward_type == "log":
            reward = 100 * (- np.log(0.5 * abs(theta1) + np.exp(1) - np.pi / 8) + 1)

        elif reward_type == "penalty":
            # Pure negative rewards
            reward = - abs(theta1)/120

        elif reward_type == "inverse_penalty":
            # Pure negative rewards
            # To be used when 'inverse' states are used
            reward = - (120 - abs(theta1))

        elif reward_type == "point_reached":
            if self.way_point_reached_in_step:
                reward = 100
            else:
                reward = 0

        if self.way_point_reached_in_step:
            reward += 100

        return reward

########################################################################################################################
# -------------------------------------------RUN AND RENDER FUNCTIONS------------------------------------------------- #
########################################################################################################################

    # @jit()
    def step(self, action: int):
        # TODO: Fix step function to iterate a few times before returning
        if self.terminal:
            logging.error("Terminal state has been reached. Please call sim.reset() in order to restart the simulation")
            return None
        if not self.has_been_reset:
            logging.error("Please call .reset() before calling .step()")
            return None

        # Actions are encoded as:
        # 0 = +10 degrees left
        # 1 = +5 degrees left
        # 2 = +3 degrees left
        # 3 = +1 degrees left
        # 4 = 0 degrees
        # 5 = +1 degrees right
        # 6 = +3 degrees right
        # 7 = +5 degrees right
        # 8 = *10 degrees right

        # if action == 0:
        #     steer_angle = 10
        # elif action == 1:
        #     steer_angle = 5
        # elif action == 2:
        #     steer_angle = 3
        # elif action == 3:
        #     steer_angle = 1
        # elif action == 4:
        #     steer_angle = 0
        # elif action == 5:
        #     steer_angle = -1
        # elif action == 6:
        #     steer_angle = -3
        # elif action == 7:
        #     steer_angle = -5
        # elif action == 8:
        #     steer_angle = -10

        if action == 0:
            steer_angle = - 5
        elif action == 1:
            steer_angle = 0
        elif action == 2:
            steer_angle = 5
        else:
            print("Invalid action")
            return None

        angle_increment = (np.deg2rad(steer_angle) - self.vehicle.delta)/10
        # angle_increment = np.deg2rad(steer_angle / 10)
        set_angle = self.vehicle.delta
        for iteration in range(self.iterations_per_step):
            if iteration < 10:
                set_angle += angle_increment

            vehicle_status = self.vehicle.drive(steering_angle=set_angle)
            vehicle_coords = vehicle_status[0:2]
            self._vehicle_coords.append(vehicle_coords)
            # print(f"Vehicle status: {vehicle_status}")
            vehicle_heading = vehicle_status[2]
            self._get_state(vehicle_coords=vehicle_coords, vehicle_heading=vehicle_heading)
            self.run_time += self.dt

            end_condition = ""

            if self.iterations_without_progress >= 500:
                self.terminal = True
                end_condition = "No progress"
                break

            if self.run_time >= self.timeout:
                self.terminal = True
                end_condition = "Timeout"
                break

            if abs(self.last_state[2]) >= 120:
                self.terminal = True
                end_condition = "Error exceeded 120 degrees"
                break

        reward = self._calculate_reward(reward_type="penalty")

        if self.terminal:
            reward -= 1000

        self.results.append([self.current_run, self.last_state, reward, self.points_reached, self.terminal,
                             self.run_time, end_condition])

        return self.StepReturn(state=self.last_state, reward=reward, progress=self.points_reached,
                               terminal=self.terminal, run_time=self.run_time, end_condition=end_condition)

########################################################################################################################
# -------------------------------------LOGGING AND RENDERING FUNCTIONS------------------------------------------------ #
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

    def log_data(self, file_name: str):
        # vehicle_data = self.vehicle.parameter_history()
        # tyre_data = self.vehicle.tyre.parameter_history()
        simulation_data = pd.DataFrame(self.results, columns=["Run", "State", "Reward", "Points Reached",
                                                              "Terminal", "Run Time", "End Condition"])

        simulation_data.to_csv(f"./Results/{file_name}.csv", sep=',')

    # RENDERING
    def _setup_rendering(self):
        x_max = np.max(self.input_data[:, 0])
        x_min = np.min(self.input_data[:, 0])
        y_max = np.max(self.input_data[:, 1])
        y_min = np.min(self.input_data[:, 1])

        x_range = x_max - x_min
        y_range = y_max - y_min

        available_x_space = self.SIZE[0] - 250
        available_y_space = self.SIZE[1] - 50

        # Set Scaling Factors
        x_scale = available_x_space / x_range
        y_scale = available_y_space / y_range

        if x_scale >= y_scale:
            self._scaling_factor = y_scale
        else:
            self._scaling_factor = x_scale

        self._way_point_render_radius = int(round(self.way_point_threshold * self._scaling_factor))

        # Set Origins
        data_origin = self.input_data[0]
        x_distance_to_min = data_origin[0] - x_min
        y_distance_to_max = y_max - data_origin[1]

        self._path_origin = (25 + int(round(x_distance_to_min * self._scaling_factor)),
                             25 + int(round(y_distance_to_max * self._scaling_factor)))

        self._text_box = pygame.Rect(self.SIZE[0] - 200, 0, 200, self.SIZE[1])

        # Create Display and Load Sprites
        self._screen = pygame.display.set_mode(self.SIZE)
        self._vehicle_sprite = pygame.image.load("res/SoftTarget.png")

        vehicle_length = int(round(2 * self._scaling_factor))
        vehicle_width = int(round(2 * self._scaling_factor))
        self._vehicle_sprite = pygame.transform.scale(self._vehicle_sprite, (vehicle_length, vehicle_width))
        self._vehicle_sprite_rect = self._vehicle_sprite.get_rect()

        self._heading_font = pygame.font.SysFont('arial', 20)
        self._data_font = pygame.font.SysFont('arial', 10)

    def _rendering_coordinate_conversion(self, coords):

        coord_array = np.array(coords)

        if coord_array.shape == (2,):
            coord_array = np.expand_dims(coord_array, axis=0)

        x = [int(round(self._path_origin[0] + self._scaling_factor * coord[0])) for coord in coord_array]
        y = [int(round(self._path_origin[1] - self._scaling_factor * coord[1])) for coord in coord_array]
        new_coords = list(zip(x, y))

        if len(new_coords) == 1:
            return new_coords[0]

        return new_coords

    def _draw_data(self, vehicle_heading, vehicle_delta):
        # Draw Axes
        pygame.draw.line(self._screen, (0, 0, 0), tuple(np.subtract(self._path_origin, [25, 0])),
                         tuple(np.add(self._path_origin, [50, 0])), 1)
        pygame.draw.line(self._screen, (0, 0, 0), tuple(np.subtract(self._path_origin, (0, 25))),
                         tuple(np.add(self._path_origin, (0, 50))), 1)

        # Draw Waypoints
        for point in self.input_data:
            pygame.draw.circle(self._screen, (10, 200, 10), self._rendering_coordinate_conversion(point), 2)

            pygame.draw.circle(self._screen, (200, 10, 10), self._rendering_coordinate_conversion(point),
                               self._way_point_render_radius, 1)

        # Draw Vehicle Data
        if len(self._vehicle_coords) >= 2:
            pygame.draw.lines(self._screen, (0, 0, 0), False,
                              self._rendering_coordinate_conversion(self._vehicle_coords), 1)

        # Draw textbox and text
        # Text Heading
        text_heading = self._heading_font.render("VEHICLE DATA", True, (0, 0, 0))
        text_heading_rect = text_heading.get_rect()
        text_heading_lefttop = (self._text_box.centerx - int(round(text_heading_rect.width / 2) + 20), 50)
        # Heading Label
        heading_label = self._data_font.render("VEHICLE HEADING", True, (0, 0, 0))
        heading_label_rect = heading_label.get_rect()
        heading_label_lefttop = (self._text_box.centerx - int(round(heading_label_rect.width / 2) + 20), 100)
        # Heading Data
        heading_data = self._data_font.render(f"{np.degrees(vehicle_heading)}", True, (200, 10, 10))
        heading_data_rect = heading_data.get_rect()
        heading_data_lefttop = (self._text_box.centerx - int(round(heading_data_rect.width / 2) + 20), 120)
        # Steering Label
        steering_label = self._data_font.render("VEHICLE STEERING ANGLE", True, (0, 0, 0))
        steering_label_rect = steering_label.get_rect()
        steering_label_lefttop = (self._text_box.centerx - int(round(steering_label_rect.width / 2 + 20)), 145)
        # Steering Data
        steering_data = self._data_font.render(f"{round(np.rad2deg(vehicle_delta))}", True, (200, 10, 10))
        steering_data_rect = steering_data.get_rect()
        steering_data_lefttop = (self._text_box.centerx - int(round(steering_data_rect.width / 2) + 20), 165)
        # Heading Error 1 Label
        heading_er_1_label = self._data_font.render("HEADING ERROR TO P1", True, (0, 0, 0))
        heading_er_1_label_rect = heading_er_1_label.get_rect()
        heading_er_1_label_lefttop = (self._text_box.centerx - int(round(heading_er_1_label_rect.width / 2 + 20)), 190)
        # Heading Error 1 Data
        heading_er_1_data = self._data_font.render(f"{round(np.rad2deg(self.last_state[2]))}", True, (200, 10, 10))
        heading_er_1_data_rect = heading_er_1_data.get_rect()
        heading_er_1_data_lefttop = (self._text_box.centerx - int(round(heading_er_1_data_rect.width / 2) + 20), 210)
        # Heading Error 2 Label
        heading_er_2_label = self._data_font.render("HEADING ERROR TO P2", True, (0, 0, 0))
        heading_er_2_label_rect = heading_er_2_label.get_rect()
        heading_er_2_label_lefttop = (self._text_box.centerx - int(round(heading_er_2_label_rect.width / 2 + 20)), 235)
        # Heading Error 2 Data
        heading_er_2_data = self._data_font.render(f"{round(np.rad2deg(self.last_state[3]))}", True, (200, 10, 10))
        heading_er_2_data_rect = heading_er_2_data.get_rect()
        heading_er_2_data_lefttop = (self._text_box.centerx - int(round(heading_er_2_data_rect.width / 2) + 20), 255)
        # Distance 1 Label
        distance_1_label = self._data_font.render("DISTANCE TO P1", True, (0, 0, 0))
        distance_1_label_rect = distance_1_label.get_rect()
        distance_1_label_lefttop = (self._text_box.centerx - int(round(distance_1_label_rect.width / 2 + 20)), 280)
        # Distance 1 Data
        distance_1_data = self._data_font.render(f"{round(self.last_state[0], 2)}", True, (200, 10, 10))
        distance_1_data_rect = distance_1_data.get_rect()
        distance_1_data_lefttop = (self._text_box.centerx - int(round(distance_1_data_rect.width / 2) + 20), 300)
        # Distance 2 Label
        distance_2_label = self._data_font.render("DISTANCE TO P2", True, (0, 0, 0))
        distance_2_label_rect = distance_2_label.get_rect()
        distance_2_label_lefttop = (self._text_box.centerx - int(round(distance_2_label_rect.width / 2 + 20)), 325)
        # Distance 2 Data
        distance_2_data = self._data_font.render(f"{round(self.last_state[1], 2)}", True, (200, 10, 10))
        distance_2_data_rect = distance_2_data.get_rect()
        distance_2_data_lefttop = (self._text_box.centerx - int(round(distance_2_data_rect.width / 2) + 20), 345)
        # Points Reached Label
        points_reached_label = self._data_font.render("POINTS REACHED", True, (0, 0, 0))
        points_reached_label_rect = points_reached_label.get_rect()
        points_reached_label_lefttop = (self._text_box.centerx - int(round(points_reached_label_rect.width / 2 + 20)), 370)
        # Distance 2 Data
        points_reached_data = self._data_font.render(f"{self.points_reached}", True, (200, 10, 10))
        points_reached_data_rect = points_reached_data.get_rect()
        points_reached_data_lefttop = (self._text_box.centerx - int(round(points_reached_data_rect.width / 2) + 20), 390)

        self._screen.blit(text_heading, text_heading_lefttop)

        self._screen.blit(heading_label, heading_label_lefttop)
        self._screen.blit(heading_data, heading_data_lefttop)

        self._screen.blit(steering_label, steering_label_lefttop)
        self._screen.blit(steering_data, steering_data_lefttop)

        self._screen.blit(heading_er_1_label, heading_er_1_label_lefttop)
        self._screen.blit(heading_er_1_data, heading_er_1_data_lefttop)

        self._screen.blit(heading_er_2_label, heading_er_2_label_lefttop)
        self._screen.blit(heading_er_2_data, heading_er_2_data_lefttop)

        self._screen.blit(distance_1_label, distance_1_label_lefttop)
        self._screen.blit(distance_1_data, distance_1_data_lefttop)

        self._screen.blit(distance_2_label, distance_2_label_lefttop)
        self._screen.blit(distance_2_data, distance_2_data_lefttop)

        self._screen.blit(points_reached_label, points_reached_label_lefttop)
        self._screen.blit(points_reached_data, points_reached_data_lefttop)

        pygame.draw.rect(self._screen, (0, 0, 0), self._text_box, 1)

    def render(self):
        if self._screen is None:
            self._setup_rendering()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Simulation stopped. Exiting.")
                sys.exit()

        vehicle_status = self.vehicle.get_status()
        vehicle_delta = self.vehicle.delta

        # Clear Screen
        self._screen.fill((255, 255, 255))
        # Draw axes, points etc.
        self._draw_data(vehicle_heading=vehicle_status[2], vehicle_delta=vehicle_delta)

        # Draw vehicle heading
        # First calculates scaled angle
        x = np.cos(vehicle_status[2])
        y = np.sin(vehicle_status[2])
        theta_prime = np.arctan2(y * self._scaling_factor, x * self._scaling_factor)

        vehicle_pos = self._rendering_coordinate_conversion(vehicle_status[:2])
        end = [vehicle_pos[0] + 25 * np.cos(theta_prime), vehicle_pos[1] - 25 * np.sin(theta_prime)]
        pygame.draw.line(self._screen, (10, 10, 200), vehicle_pos, end, 1)

        # Draw vehicle steering angle
        # First calculates scaled angle
        x = np.cos(vehicle_status[2] + vehicle_delta)
        y = np.sin(vehicle_status[2] + vehicle_delta)
        delta_prime = np.arctan2(y * self._scaling_factor, x * self._scaling_factor)

        end = [vehicle_pos[0] + round(25 * np.cos(delta_prime)), vehicle_pos[1] - round(25 * np.sin(delta_prime))]
        pygame.draw.line(self._screen, (10, 200, 10), vehicle_pos, end, 1)

        # Render Vehicle
        # Rotate Image (return copy)
        rotated_vehicle_sprite = pygame.transform.rotate(self._vehicle_sprite, np.degrees(theta_prime))

        self._vehicle_sprite_rect = rotated_vehicle_sprite.get_rect()
        self._vehicle_sprite_rect.center = vehicle_pos
        self._screen.blit(rotated_vehicle_sprite, self._vehicle_sprite_rect)

        # Display everything
        pygame.display.flip()