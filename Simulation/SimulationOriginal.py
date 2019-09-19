from VehicleModel.Vehicle import Vehicle
from Controller.DQNController import DQNController

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pandas as pd
import numpy as np
import logging
from numba import jit
from typing import Optional, List
import time

import yaml


class Simulation:

    # logging.basicConfig(format='%(levelname)s-%(message)s')

    controller: DQNController
    vehicle: Vehicle

    # Simulation parameters
    debug: bool
    dt: float     # Time step [s]

    style.use('fast')

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)

    def __init__(self, config: dict):
        try:
            self.dt = config["simulation-parameters"]["timestep"]
            self.input_type = config["simulation-parameters"]["input-type"]
            self.input_file = config["simulation-parameters"]["input"]
            self.debug = config["simulation-parameters"]["debug"]
            name = self.input_file.split("/")[-1]
            name = name.replace(".txt", "")
            t = time.strftime("%H-%M-%S", time.gmtime())
            self.simulation_name = f"{name}_{t}"
            self.run_time = 0.
            self.results: Optional[pd.DataFrame] = None
            self.input_data = self._read_input()

            if self.input_type == "path":
                self.closest_points = np.array([self.input_data[0:3], 0])
                x0, y0, theta0 = self._define_initial_conditions_if_path()
                self.vehicle = Vehicle(vehicle_config=config["vehicle-parameters"], tyre_config=config["tyre-model"],
                                       dt=self.dt, x0=x0, y0=y0, theta0=theta0)
            elif self.input_type == "steering":
                self.vehicle = Vehicle(vehicle_config=config["vehicle-parameters"], tyre_config=config["tyre-model"],
                                       dt=self.dt)

            # if config["controller"]["type"] == "DQN":

            self.controller = DQNController(observation_space=4, action_space=2)

        except KeyError as e:
            logging.error("Config file missing/incorrect")
            print("Key Error")
            logging.error(e)
            exit(1)

    def run(self, dt: float = None, timeout: float = 50., live_plot: bool = False) -> List[pd.DataFrame]:
        if dt is not None:
            self.dt = dt

        if self.input_type == "path":
            self.results = self._run_path(timeout=timeout)
            print(f"Running path for {self.input_file}")

        elif self.input_type == "steering":
            self.results = self._run_steering()
            print(f"Running steering for {self.input_file}")

        else:
            logging.error(f"Incorrect simulation type"
                          f" (Type: {self.input_type}) selected. Ensure config file is correct")
            exit(1)

        self.log_error_information()
        return [self.results, self.vehicle.tyre.parameter_history(), self.controller.parameter_history()]

    def _run_path(self, timeout: float) -> pd.DataFrame:
        end_of_path = False
        path_reached = False
        while self.run_time <= timeout and not path_reached:
            # Retrieve vehicle data
            px = np.array([self.vehicle.global_x, self.vehicle.global_y])
            vehicle_theta = self.vehicle.global_theta

            # Update calculations points if needed
            if not end_of_path:
                end_of_path = self._update_closest_points(px=px)
            else:
                logging.warning("End of path reached")
                p2 = self.closest_points[0][1]
                if np.sqrt((px[0] - p2[0]) ** 2 + (px[1] - p2[1]) ** 2) <= 0.1:
                    path_reached = True

            # Calculate lateral and yaw errors
            errors = self._calculate_errors(px=px, vehicle_theta=vehicle_theta)

            # print(f"Times : {self.run_time} || Lateral Error : {errors[0]} || Yaw Error : {errors[1]}")

            # Use controller to calculate steering output
            steering = self.controller.calculate_steering(errors=errors, dt=self.dt)

            self.vehicle.drive(steering_angle=steering)

            self.run_time += self.dt
            # logging.warning(f"Current timestep {self.run_time}")

        return self.vehicle.parameter_history()

    def _run_steering(self) -> pd.DataFrame:
        for angle in self.input_data:
            self.vehicle.drive(steering_angle=angle)
            self.run_time += self.dt

        return self.vehicle.parameter_history()

    def _define_initial_conditions_if_path(self) -> float(3):
        p1 = self.closest_points[0][0]
        p2 = self.closest_points[0][1]

        x = p1[0]
        y = p1[1]
        theta = self._calculate_path_angle(p1, p2)

        return x, y, theta

    def _calculate_path_angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        theta = np.arctan2(delta_y, delta_x)
        return theta

    def _update_closest_points(self, px: np.ndarray):
        """
        Calculates the new path points closest to the vehicle. Retains the order of the path, thus if a point further
        down the path appears closer to the vehicle, it will be ignored
        :param px: np.ndarray() : Current vehicle x and y coords.
        :return: None
        """

        p1 = self.closest_points[0][0]
        p3 = self.closest_points[0][2]

        dist_to_1 = np.sqrt((px[0] - p1[0])**2 + (px[1] - p1[1])**2)
        dist_to_3 = np.sqrt((px[0] - p3[0]) ** 2 + (px[1] - p3[1]) ** 2)

        if dist_to_3 <= dist_to_1:
            i = self.closest_points[1] + 1
            self.closest_points[0] = self.input_data[i: i+3]
            self.closest_points[1] = i

        if len(self.closest_points[0]) < 3:
            logging.warning("Closest points less than 3")
            return True

        else:
            return False

    def _calculate_errors(self, px: np.ndarray, vehicle_theta: float) -> np.ndarray:
        """
        Calculates the lateral and yaw errors of the vehicle w.r.t. the current path points.
        :param px: np.ndarray : Current vehicle x and y coords
        :param vehicle_theta: float : Current heading of the vehicle
        :return: np.ndarray : [lateral error, yaw error]
        """

        p1 = self.closest_points[0][0]
        p2 = self.closest_points[0][1]

        # Yaw error
        path_theta = self._calculate_path_angle(p1, p2)
        yaw_err = path_theta - vehicle_theta

        # Lateral error
        lat_err = np.cross(p2 - p1, px - p1) / np.linalg.norm(p2 - p1)

        # Determine sign of lat error
        vehicle_angle_from_p1 = self._calculate_path_angle(p1, px)
        if vehicle_angle_from_p1 < path_theta:
            lat_err = lat_err * -1

        return np.array([lat_err, yaw_err])

    ########################################################

    def _read_input(self) -> np.ndarray:
        data = []
        try:
            with open(self.input_file) as file:
                for line in file:
                    if self.input_type == "path":
                        vals = line.split(',')
                        data.append([float(vals[0]), float(vals[1])])
                    else:
                        data.append(float(line))

            return np.array(data)
        except ValueError as e:
            logging.error(f"Input file {self.input_file} incorrect")
            logging.error(e)
            exit(1)
        except FileNotFoundError as e:
            logging.error(f"Input file {self.input_file} does not exist")
            logging.error(e)
            exit(1)

############################################################################################
    # PLOTS

    def plot_vehicle_values(self, values: list, x_label: str, y_label: str, title: str):
        plt.figure(figsize=(10, 8), dpi=100)
        vehicle_data = self.vehicle.parameter_history()
        for value in values:
            try:
                data = vehicle_data[value]
                plt.plot(vehicle_data["Time"], data, label=value)
            except KeyError as e:
                logging.error(f"Incorrect key: {value} does not matched obtained data. Skipping this value")

        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        # plt.savefig(f"Results/{title}.png")
        plt.show()

    def plot_controller_values(self, values: list, x_label: str, y_label: str, title: str):
        plt.figure(figsize=(10, 8), dpi=100)
        controller_data = self.controller.parameter_history()
        for value in values:
            try:
                data = controller_data[value]
                plt.plot(controller_data["Time"], data, label=value)
            except KeyError as e:
                logging.error(f"Incorrect key: {value} does not matched obtained data. Skipping this value")

        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        # plt.savefig(f"Results/{title}.png")
        plt.show()

    def plot_position(self):
        plt.figure(figsize=(10, 8), dpi=100)
        vehicle_data = self.vehicle.parameter_history()
        plt.plot(vehicle_data["x"], vehicle_data["y"], label="Vehicle Path")
        plt.title("Vehicle Path")

        if self.input_type == "path":
            plt.plot(self.input_data[:, 0], self.input_data[:, 1], label="Input Path")
            plt.title("Vehicle Path vs Input Path")

        plt.legend()
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        # plt.savefig(f"Results/Vehicle_Position.png")
        plt.show()

    def animate(self):
        # TODO: Add animated plot
        vehicle_parameters = self.vehicle.parameter_history()

        xs = vehicle_parameters["x"]
        ys = vehicle_parameters["y"]

        self.ax1.clear()
        self.ax1.plot(xs, ys)

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

