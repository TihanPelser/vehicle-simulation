from controller.DQNControllerModel import DQNController
from simulation.waypoint_simulation import WayPointSimulation
from simulation.path_simulation import PathSimulation
from vehicle_models.dynamic_model import DynamicVehicleModel
from vehicle_models.kinematic_model import KinematicVehicleModel
from tyre_model.LinearCutoff import LinearTyre
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TIME_STEP = 0.001
DATA_FILES = ["sine_continuous_xy_rotated.txt"]
LOG_FILE = "LC_TEST_PATH_SINE"


def read_data(file_name: str):
    file_data = []
    with open(f"TrainingData/{file_name}", "r") as file:
        for line in file:
            vals = line.split(',')
            file_data.append([float(vals[0]), float(vals[1])])
    return np.array(file_data)


def log_data(x, y, lat_err, yaw_err, action):
    with open(f"ComparisonTests/{LOG_FILE}.txt", "a+") as file:
        file.write(f"{x},{y},{lat_err},{yaw_err},{action}\n")


if __name__ == "__main__":
    
    data = read_data(DATA_FILES[0])

    dqn = DQNController(model_file="models/2_IN_5_OUT_PATH_FINAL.h5")
    vehicle_kinematic = KinematicVehicleModel(dt=TIME_STEP)

    simulation = PathSimulation(sim_name="Testing", vehicle=vehicle_kinematic, input_data=data,
                                time_step=TIME_STEP, timeout=60., iterations_per_step=50, way_point_threshold=0.5,
                                distance_between_points=5.)
    action_space = 5
    observation_space = 2

    run = 0
    try:
        while True:
            run += 1
            state = simulation.reset(epsilon=0)
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
                print(action)
                # state_next, reward, points_reached, terminal, time = simulation.step(step_type="action", input=action)
                step_time_taken = time.time()
                results = simulation.step(action=action)
                step_time += time.time() - step_time_taken

                state_next = results.state
                reward = results.reward
                terminal = results.terminal
                points_reached = results.progress
                run_time = results.run_time
                end_condition = results.end_condition

                # reward = reward if not terminal else -reward
                log_data(x=vehicle_kinematic.global_x, y=vehicle_kinematic.global_y, lat_err=state[0, 0],
                         yaw_err=state[0, 1], action=action)

                total_reward += reward
                # print(f"State next: {state_next}")
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

