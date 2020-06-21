from controller.dqn_controller import DQNController
from simulation.path_simulation import PathSimulation
from vehicle_models.kinematic_model import KinematicVehicleModel
from tyre_model.LinearCutoff import LinearTyre

import numpy as np
import time

TIME_STEP = 0.001
DATA_FILE = "sine_continuous_xy_rotated.txt"
LOG_FILE = "LC_TEST_PATH_SINE"
SIMULATION_NAME = "test_sim"
MODEL_FILE = "models/final_model.h5"


def read_data(file_name: str):
    file_data = []
    with open(f"TrainingData/{file_name}", "r") as file:
        for line in file:
            vals = line.split(',')
            file_data.append([float(vals[0]), float(vals[1])])
    return np.array(file_data)


def scale_state(new_state: np.ndarray) -> np.ndarray:
    bounded_lateral_error = new_state[0] / 2.5
    bounded_yaw_error = new_state[1] / np.pi
    return np.array([bounded_lateral_error, bounded_yaw_error])


def log_data(x_pos: float, y_pos: float, lat_err: float, yaw_err: float, action_taken: int):
    with open(f"ComparisonTests/{LOG_FILE}.txt", "a+") as file:
        file.write(f"{x_pos},{y_pos},{lat_err},{yaw_err},{action_taken}\n")


if __name__ == "__main__":
    
    data = read_data(DATA_FILE)

    dqn = DQNController.load(model_file=MODEL_FILE,
                             action_space_size=5,
                             observation_space_size=2)

    action_space = 5
    observation_space = 2
    run = 0

    tyre_model = LinearTyre()

    vehicle_kinematic = KinematicVehicleModel(dt=TIME_STEP)

    simulation = PathSimulation(sim_name=SIMULATION_NAME,
                                vehicle=vehicle_kinematic,
                                input_data=data,
                                time_step=TIME_STEP,
                                timeout=60.,
                                iterations_per_step=50,
                                way_point_threshold=0.5,
                                distance_between_points=5.)

    try:
        while True:
            run += 1
            state = simulation.reset(epsilon=0)

            state = scale_state(state)
            state = np.reshape(state, [1, observation_space])
            step = 0
            total_reward = 0
            start = time.time()

            train_time = 0
            step_time = 0

            while True:
                step += 1
                simulation.render()
                action = dqn.act(state)
                step_time_taken = time.time()
                results = simulation.step(action=action)
                step_time += time.time() - step_time_taken

                state_next = results.state
                reward = results.reward
                terminal = results.terminal
                points_reached = results.progress
                run_time = results.run_time
                end_condition = results.end_condition

                log_data(x_pos=vehicle_kinematic.global_x, y_pos=vehicle_kinematic.global_y, lat_err=state[0, 0],
                         yaw_err=state[0, 1], action_taken=action)

                total_reward += reward

                state_next = scale_state(state_next)

                state_next = np.reshape(state_next, [1, observation_space])
                state = state_next

                if terminal:
                    print("\n===================================================")
                    print("PREVIOUS RUN DATA")
                    print("===================================================")
                    print(f"Run: {run}, exploration: {dqn.exploration_rate}, steps: {step}")
                    print(f"Total sim time : {run_time}, Total points reached: {points_reached}")
                    print(f"Run end condition: {end_condition}")
                    print(f"Total rewards: {total_reward}")
                    print("===================================================\n")
                    break

            print("\n===================================================")
            print("ALL TIME DATA")
            print("===================================================")
            print(f"Episode took {time.time() - start} seconds")
            print(f"Total step time: {step_time}")
            print("\n===================================================\n")
            break

    except KeyboardInterrupt:
        print("User exit")
        print(f"Exited on run: {run}")
