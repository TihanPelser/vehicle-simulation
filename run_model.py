from Controller.DQNController import DQNController
from Simulation.WayPointSimulation import WayPointSimulation
from Simulation.PathSimulation import PathSimulation
from VehicleModel.DynamicModel import DynamicVehicleModel
from VehicleModel.KinematicModel import KinematicVehicleModel
from TyreModel.LinearCutoff import LinearTyre
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TIME_STEP = 0.001
DATA_FILES = ["DLC.txt", "Sine.txt"]


def read_data(file_name: str):
    file_data = []
    with open(f"TrainingData/{file_name}", "r") as file:
        for line in file:
            vals = line.split(',')
            file_data.append([float(vals[0]), float(vals[1])])
    return np.array(file_data)

def bound_state(state: np.ndarray) -> np.ndarray:
    bounded_lateral_error = state[0] / 2.5
    bounded_yaw_error = state[1] / np.pi
    return np.array([bounded_lateral_error, bounded_yaw_error])


if __name__ == "__main__":

    action_space = 5
    observation_space = 2

    data = read_data(DATA_FILES[1])

    dqn = DQNController(action_space=action_space, observation_space=observation_space, gpu_count=0, cpu_count=16, check_name="Pretrained")
    dqn.model.load_weights("Checkpoints/2_IN_5_OUT_PATH.ckpt")
    tyre_model = LinearTyre()
    vehicle_kinematic = KinematicVehicleModel(dt=TIME_STEP)
    # vehicle_dynamic = DynamicVehicleModel()

    simulation = PathSimulation(sim_name="Training1", vehicle=vehicle_kinematic, input_data=data,
                                    time_step=TIME_STEP, timeout=60., iterations_per_step=50, way_point_threshold=0.5,
                                    distance_between_points=5.)


    run = 0
    dqn.model.save("2_IN_5_OUT_PATH_FINAL.h5")
    try:
        while True:
            run += 1
            state = simulation.reset(epsilon=0)
            state = bound_state(state)
            state = np.reshape(state, [1, observation_space])
            step = 0
            total_reward = 0
            start = time.time()

            train_time = 0
            step_time = 0

            while True:
                step += 1
                simulation.render()
                action = dqn.act_no_explore(state)
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

                total_reward += reward
                # print(f"State next: {state_next}")
                state_next = bound_state(state_next)
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

    except KeyboardInterrupt:
        print("User exit")
        print(f"Exited on run: {run}")
