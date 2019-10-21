from Controller.DQNController import DQNController
from Simulation.PathSimulation import PathSimulation
from VehicleModel.DynamicModel import DynamicVehicleModel
from VehicleModel.KinematicModel import KinematicVehicleModel
from TyreModel.LinearCutoff import LinearTyre
import numpy as np
import time
import pandas as pd

TRAINING_FILES = ["DLC.txt", "Sine.txt"]
TIME_STEP = 0.001
SAVE_NAME = "2_IN_5_OUT_PATH"
LIVE_PLOT = False
LOGGING_FILE = f"{SAVE_NAME}_LOG.txt"


def read_data(file_name: str):
    file_data = []
    with open(f"TrainingData/{file_name}", "r") as file:
        for line in file:
            vals = line.split(',')
            file_data.append([float(vals[0]), float(vals[1])])
    return np.array(file_data)


def log(episode_num: int, lat_errors, yaw_errors, actions, rewards):
    episode_data = np.stack((lat_errors, yaw_errors, actions, rewards), axis=-1)
    frame = pd.DataFrame(data=episode_data, columns=["Lateral Error", "Yaw Error", "Action", "Reward"])
    frame.to_csv(f"TrainingResults/{SAVE_NAME}_{episode_num}", sep=",", index=False)


def bound_state(state: np.ndarray) -> np.ndarray:
    bounded_lateral_error = state[0] / 2.5
    bounded_yaw_error = state[1] / np.pi
    return np.array([bounded_lateral_error, bounded_yaw_error])


if __name__ == "__main__":

    data = read_data(TRAINING_FILES[0])

    tyre_model = LinearTyre()
    vehicle_kinematic = KinematicVehicleModel(dt=TIME_STEP)
    # vehicle_dynamic = DynamicVehicleModel()

    simulation = PathSimulation(sim_name="PathTraining1", vehicle=vehicle_kinematic, input_data=data,
                                time_step=TIME_STEP, timeout=60., iterations_per_step=50, way_point_threshold=0.5,
                                distance_between_points=5.)
    observation_space = 2
    action_space = 5
    dqn = DQNController(observation_space=observation_space, action_space=action_space, check_name=SAVE_NAME)
    # dqn.model.load_weights("Models/Working-2-Input-5-Output.hdf5")
    episode = 0

    max_steps = 0
    max_step_run = 0
    most_points = 0
    most_points_run = 0
    has_solved = False
    solved_run = None

    cumulative_steps = 0

    try:
        while True:
            all_lat_errors = []
            all_yaw_errors = []
            all_actions = []
            all_rewards = []
            episode += 1
            state = simulation.reset(dqn.exploration_rate)
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
                all_lat_errors.append(state[0, 0])
                all_yaw_errors.append(state[0, 1])
                action = dqn.act(state)
                all_actions.append(action)

                step_time_taken = time.time()
                results = simulation.step(action=action)
                step_time += time.time() - step_time_taken

                state_next = results.state
                reward = results.reward
                terminal = results.terminal
                run_time = results.run_time
                end_condition = results.end_condition

                all_rewards.append(reward)

                total_reward += reward
                state_next = bound_state(state_next)
                state_next = np.reshape(state_next, [1, observation_space])
                dqn.remember(state=state, action=action, reward=reward, next_state=state_next, done=terminal)
                state = state_next

                if terminal:
                    cumulative_steps += step

                    print("\n===================================================")
                    print("PREVIOUS RUN DATA")
                    print("===================================================")
                    print(f"Run: {episode}, exploration: {dqn.exploration_rate}, steps: {step}")
                    print(f"Total sim time : {run_time}")
                    print(f"Run end condition: {end_condition}")
                    print(f"Total rewards: {total_reward}")
                    print("===================================================\n")

                    print("CUMULATIVE STATS")
                    print(f"Average steps: {cumulative_steps / episode}")
                    print(f"Longest run: {max_step_run} with {max_steps} steps")
                    break

                train = time.time()
                dqn.experience_replay()
                train_time += time.time() - train

            print("\n===================================================")
            print("ALL TIME DATA")
            print("===================================================")
            print(f"Episode took {time.time() - start} seconds")
            print(f"Total step time: {step_time}")
            print(f"Total training time: {train_time}")
            print("\n===================================================\n")

            if len(all_lat_errors) > 1:
                if episode == 1:
                    log(episode_num=episode, lat_errors=all_lat_errors, yaw_errors=all_yaw_errors, rewards=all_rewards,
                        actions=all_actions)

                if episode % 10 == 0:
                    log(episode_num=episode, lat_errors=all_lat_errors, yaw_errors=all_yaw_errors, rewards=all_rewards,
                        actions=all_actions)

    except KeyboardInterrupt:
        print("User exit")
        exit(0)

