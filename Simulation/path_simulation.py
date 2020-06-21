from vehicle_models.kinematic_model import KinematicVehicleModel
from vehicle_models.dynamic_model import DynamicVehicleModel
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Union
from collections import namedtuple
import time
import pygame
import sys
import random


class PathSimulation:
    # CUSTOM DATA TYPES
    #  NamedTuple to save initial setup state for reset
    Initial = namedtuple("Initial", "name input_data dt timeout iterations_per_step "
                                    "way_point_threshold distance_between_points")
    StepReturn = namedtuple("StepReturn", "state reward progress terminal run_time end_condition")

    _vehicle_coords: List = []

    # simulation parameters
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

    number_of_points: int
    prev_lat_error: Optional[float]
    prev_yaw_error: Optional[float]
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

    def __init__(self, sim_name: str, vehicle: Union[KinematicVehicleModel, DynamicVehicleModel],
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

        self.preview_distance = 1.5
        self.preview_point = [0, 0]
        self.intersect_point = [0, 0]
        self.intercept_interval_index = [0, 1]
        self.previous_lateral_error = 0.

        print("Sim created...")
        print(f"Number of Waypoints: {len(input_data)}")

    def reset(self, epsilon):

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
        self.distance_between_points = self.initial.DIST_BETWEEN_POINTS
        self.run_time = 0.
        self.results = []

        # Sim name
        name = self.initial.name
        t = time.strftime("%H-%M-%S", time.gmtime())
        base_name = f"{name}_{t}"
        self.simulation_name = f"{base_name}_{self.episode}"
        self._set_vehicle_position(random_state=True, epsilon=epsilon)

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

    def _set_vehicle_position(self, random_state: bool = False, epsilon: float = 1):
        x = self.input_data[0][0]
        y = self.input_data[0][1]
        delta_x = self.input_data[1][0] - x
        delta_y = self.input_data[1][1] - y

        heading = np.arctan2(delta_y, delta_x)
        if random_state:
            # Add random noise between -45 and 45 to initial heading
            noise = np.deg2rad((random.random() * 90) - 45)
            heading += noise * epsilon

        self.vehicle.set_position(x, y, heading)

########################################################################################################################
# -------------------------------------------CALCULATION FUNCTIONS---------------------------------------------------- #
########################################################################################################################

    def _calculate_errors(self, vehicle_coords: np.ndarray, vehicle_heading: float):
        def line(point_1, point_2):
            a = (point_1[1] - point_2[1])
            b = (point_2[0] - point_1[0])
            c = (point_1[0] * point_2[1] - point_2[0] * point_1[1])
            return a, b, -c

        def intersection(line_1, line_2):
            D = line_1[0] * line_2[1] - line_1[1] * line_2[0]
            Dx = line_1[2] * line_2[1] - line_1[1] * line_2[2]
            Dy = line_1[0] * line_2[2] - line_1[2] * line_2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return round(x, 2), round(y, 2)
            else:
                return None

        preview_delta = self.preview_distance * np.array([np.cos(vehicle_heading), np.sin(vehicle_heading)])
        preview_point = np.add(vehicle_coords, preview_delta)

        self.preview_point = preview_point

        alpha = np.pi / 2 + vehicle_heading
        lateral_extension_delta = np.array([np.cos(alpha), np.sin(alpha)])
        lateral_extension_point = np.add(preview_point, lateral_extension_delta)

        lateral_line = line(point_1=preview_point, point_2=lateral_extension_point)

        lateral_intersect_point = None
        self.intersect_point = lateral_intersect_point

        for interval_index in range(len(self.input_data) - 1):
            p1 = self.input_data[interval_index]
            p2 = self.input_data[interval_index + 1]

            test_line = line(point_1=p1, point_2=p2)

            intersect_point = intersection(line_1=lateral_line, line_2=test_line)

            # Check if lines intersect at all
            if intersect_point is None:
                continue

            # Check if lines intersect in segment end points
            if sorted([p1[0], p2[0], intersect_point[0]])[1] == intersect_point[0]:
                sorted_y = sorted([round(p1[1], 2), round(p2[1], 2), round(intersect_point[1], 2)])

                if sorted_y[1] == intersect_point[1]:
                    lateral_intersect_point = intersect_point
                    self.intersect_point = lateral_intersect_point
                    break
                else:
                    print(f"Y not matching! {sorted_y} vs {intersect_point[1]}")

            # Check if intersect point lies behind final path point
            if interval_index == len(self.input_data) - 2:
                intersect_dist = np.sqrt((intersect_point[0] - p1[0]) ** 2 + (intersect_point[1] - p1[1]) ** 2)
                final_point_dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                if intersect_dist > final_point_dist:
                    lateral_intersect_point = intersect_point
                    self.intersect_point = lateral_intersect_point
                    if intersect_dist >= 15:
                        self.terminal = True
                    break

        if lateral_intersect_point is None:

            print(f"Index = {interval_index}, P1 = {p1}, P2 = {p2}")
            return None, None

        self.intercept_interval_index = [interval_index, interval_index + 1]

        lateral_error = np.sqrt((lateral_intersect_point[0] - preview_point[0]) ** 2 +
                                (lateral_intersect_point[1] - preview_point[1]) ** 2)

        # Negative if intersect lies to the left of the vector extending in the direction of the vehicle, and positive
        # if otherwise
        ab = np.array([preview_point[0] - vehicle_coords[0], preview_point[1] - vehicle_coords[1]])
        ac = np.array([lateral_intersect_point[0] - vehicle_coords[0], lateral_intersect_point[1] - vehicle_coords[1]])
        lateral_error_direction = np.sign(np.linalg.det(np.array([ac, ab])))

        path_heading = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        heading_error = vehicle_heading - path_heading

        return lateral_error_direction * lateral_error, heading_error

    def _get_state(self, vehicle_coords: np.ndarray, vehicle_heading: float):
        distance_to_final_point = np.sqrt((vehicle_coords[0] - self.input_data[-1][0]) ** 2 +
                                          (vehicle_coords[1] - self.input_data[-1][1]) ** 2)

        if distance_to_final_point <= 1:
            self.solved = True
            self.last_state = np.array([0., 0.])
            return

        lateral_error, yaw_error = self._calculate_errors(vehicle_coords=vehicle_coords, vehicle_heading=vehicle_heading)

        if lateral_error is None or yaw_error is None:
            self.last_state = np.array([None, None])
            return

        self.last_state = np.array([lateral_error, yaw_error])

    def _calculate_reward(self, reward_type: str, ) -> int:
        # TODO: Idea: Use current lateral error instead of preview lateral error to calculate reward
        reward = 0
        lateral_error = self.last_state[0]
        yaw_error = self.last_state[1]

        # Reward based on change in lateral error
        if reward_type == "lateral-difference":
            reward = abs(self.previous_lateral_error) - abs(lateral_error) * 50
            # reward = -abs(lateral_error) + abs(self.previous_lateral_error) - abs(lateral_error) * 50
            # reward = - abs(lateral_error)
            # reward = 2.5 - abs(lateral_error)
        return reward

########################################################################################################################
# -------------------------------------------RUN AND RENDER FUNCTIONS------------------------------------------------- #
########################################################################################################################

    # @jit()
    def step(self, action: int):

        if self.terminal:
            logging.error("Terminal state has been reached. Please call sim.reset() in order to restart the simulation")
            return None

        if not self.has_been_reset:
            logging.error("Please call .reset() before calling .step()")
            return None

        # Actions are encoded as:
        # 0 = 10 degrees left
        # 1 = 5 degrees left
        # 2 = 0 degrees (straight)
        # 3 = 5 degrees right
        # 4 = 10 degrees right

        if action == 0:
            steer_angle = - 10
        elif action == 1:
            steer_angle = - 5
        elif action == 2:
            steer_angle = 0
        elif action == 3:
            steer_angle = 5
        elif action == 4:
            steer_angle = 10
        else:
            print("Invalid action")
            return None

        # angle_increment = 0.01676
        angle_increment = (np.deg2rad(steer_angle) - self.vehicle.delta) / 10
        # angle_increment_steps = (np.deg2rad(steer_angle) - self.vehicle.delta) / angle_increment
        # direction = np.sign(angle_increment_steps)
        # angle_increment = np.deg2rad(steer_angle / 10)
        set_angle = self.vehicle.delta
        end_condition = ""
        for iteration in range(self.iterations_per_step):
            if iteration < 10:
                set_angle += angle_increment
            # if iteration < abs(angle_increment_steps):
            #     set_angle += direction * angle_increment

            vehicle_status = self.vehicle.drive(steering_angle=set_angle)
            vehicle_coords = vehicle_status[0:2]
            self._vehicle_coords.append(vehicle_coords)
            # print(f"Vehicle status: {vehicle_status}")
            vehicle_heading = vehicle_status[2]
            self._get_state(vehicle_coords=vehicle_coords, vehicle_heading=vehicle_heading)
            self.run_time += self.dt

            if self.solved:
                self.terminal = True
                end_condition = "Final point reached"
                break

            if self.terminal:
                end_condition = "Lateral intersect point extends past max"
                break

            if self.run_time >= self.timeout:
                self.terminal = True
                end_condition = "Timeout"
                break

            if abs(self.last_state[1]) >= np.deg2rad(120):
                self.terminal = True
                end_condition = "Yaw error exceeded 120 degrees"
                break

            if self.last_state.any() is None:
                self.terminal = True
                end_condition = "Infinite lateral error"
                break

            if abs(self.last_state[0]) >= 10:
                self.terminal = True
                end_condition = "Lateral error exceeded 10 m"
                break

        reward = self._calculate_reward(reward_type="lateral-difference")

        if self.terminal and not self.solved:
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
                             int(round(available_y_space/2)))

        self._text_box = pygame.Rect(self.SIZE[0] - 200, 0, 200, self.SIZE[1])

        # Create Display and Load Sprites
        self._screen = pygame.display.set_mode(self.SIZE)
        self._vehicle_sprite = pygame.image.load("res/SoftTarget.png")

        vehicle_length = int(round(1.5 * self._scaling_factor))
        vehicle_width = int(round(1.5 * self._scaling_factor))
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

        # Lateral Error Label
        lat_err_label = self._data_font.render("LATERAL ERROR", True, (0, 0, 0))
        lat_err_label_rect = lat_err_label.get_rect()
        lat_err_label_lefttop = (self._text_box.centerx - int(round(lat_err_label_rect.width / 2 + 20)), 190)
        # Lateral Error Data
        lat_err_data = self._data_font.render(f"{round(self.last_state[0], 2)}", True, (200, 10, 10))
        lat_err_data_rect = lat_err_data.get_rect()
        lat_err_data_lefttop = (self._text_box.centerx - int(round(lat_err_data_rect.width / 2) + 20), 210)
        # Yaw Label
        yaw_err_label = self._data_font.render("YAW ERROR", True, (0, 0, 0))
        yaw_err_label_rect = yaw_err_label.get_rect()
        yaw_err_label_lefttop = (self._text_box.centerx - int(round(yaw_err_label_rect.width / 2 + 20)), 235)
        # Yaw Data
        yaw_err_data = self._data_font.render(f"{round(np.rad2deg(self.last_state[1]), 2)}", True, (200, 10, 10))
        yaw_err_data_rect = yaw_err_data.get_rect()
        yaw_err_data_lefttop = (self._text_box.centerx - int(round(yaw_err_data_rect.width / 2) + 20), 255)

        self._screen.blit(text_heading, text_heading_lefttop)

        self._screen.blit(heading_label, heading_label_lefttop)
        self._screen.blit(heading_data, heading_data_lefttop)

        self._screen.blit(steering_label, steering_label_lefttop)
        self._screen.blit(steering_data, steering_data_lefttop)

        self._screen.blit(lat_err_label, lat_err_label_lefttop)
        self._screen.blit(lat_err_data, lat_err_data_lefttop)

        self._screen.blit(yaw_err_label, yaw_err_label_lefttop)
        self._screen.blit(yaw_err_data, yaw_err_data_lefttop)

        pygame.draw.rect(self._screen, (0, 0, 0), self._text_box, 1)

    def render(self):
        if self._screen is None:
            self._setup_rendering()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("simulation stopped. Exiting.")
                sys.exit()

        vehicle_status = self.vehicle.get_status()
        vehicle_delta = self.vehicle.delta

        # Clear Screen
        self._screen.fill((255, 255, 255))
        # Draw axes, points etc.
        self._draw_data(vehicle_heading=vehicle_status[2], vehicle_delta=vehicle_delta)

        # Draw vehicle heading
        # First calculates the scaled angle
        x = np.cos(vehicle_status[2])
        y = np.sin(vehicle_status[2])
        theta_prime = np.arctan2(y * self._scaling_factor, x * self._scaling_factor)

        vehicle_pos = self._rendering_coordinate_conversion(vehicle_status[:2])
        # end = [vehicle_pos[0] + self.preview_distance * np.cos(theta_prime),
        #        vehicle_pos[1] - self.preview_distance * np.sin(theta_prime)]
        # pygame.draw.line(self._screen, (10, 10, 200), vehicle_pos, end, 1)

        # Draw vehicle steering angle
        # First calculates the scaled angle
        x = np.cos(vehicle_status[2] + vehicle_delta)
        y = np.sin(vehicle_status[2] + vehicle_delta)
        delta_prime = np.arctan2(y * self._scaling_factor, x * self._scaling_factor)

        end = [vehicle_pos[0] + round(25 * np.cos(delta_prime)), vehicle_pos[1] - round(25 * np.sin(delta_prime))]
        pygame.draw.line(self._screen, (10, 200, 10), vehicle_pos, end, 1)

        # Draw preview point and lateral intersect
        converted_preview = self._rendering_coordinate_conversion(self.preview_point)
        pygame.draw.line(self._screen, (100, 100, 200), vehicle_pos, converted_preview, 1)

        if self.intersect_point is not None:
            converted_intersect = self._rendering_coordinate_conversion(self.intersect_point)
            pygame.draw.line(self._screen, (150, 150, 200), converted_preview, converted_intersect, 1)

            # Draw intercept range
            p1 = self.input_data[self.intercept_interval_index[0]]
            p2 = self.input_data[self.intercept_interval_index[1]]
            p1 = self._rendering_coordinate_conversion(p1)
            p2 = self._rendering_coordinate_conversion(p2)
            pygame.draw.line(self._screen, (100, 200, 100), p1, p2, 1)

        # Render Vehicle
        # Rotate Image (return copy)
        rotated_vehicle_sprite = pygame.transform.rotate(self._vehicle_sprite, np.degrees(theta_prime))

        self._vehicle_sprite_rect = rotated_vehicle_sprite.get_rect()
        self._vehicle_sprite_rect.center = vehicle_pos
        self._screen.blit(rotated_vehicle_sprite, self._vehicle_sprite_rect)

        # Display everything
        pygame.display.flip()
