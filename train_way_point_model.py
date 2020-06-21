from controller.DQNController import DQNController
from simulation.waypoint_simulation import WayPointSimulation
from vehicle_models.dynamic_model import DynamicVehicleModel
from vehicle_models.kinematic_model import KinematicVehicleModel
from tyre_model.LinearCutoff import LinearTyre
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TRAINING_FILES = ["DLC.txt", "Sine.txt"]
TIME_STEP = 0.001
SAVE_NAME = "COMPLEX_ARCH_1_IN_5_OUT"
LIVE_PLOT = False
LOGGING_FILE = f"{SAVE_NAME}_LOG.txt"


def read_data(file_name: str):
    file_data = []
    with open(f"TrainingData/{file_name}", "r") as file:
        for line in file:
            vals = line.split(',')
            file_data.append([float(vals[0]), float(vals[1])])
    return np.array(file_data)


def log(run, steps, points, reward):
    with open(LOGGING_FILE, "a") as file:
        file.write(f"{run}\t{steps}\t{points}\t{reward}\n")


if __name__ == "__main__":

    data = read_data(TRAINING_FILES[1])

    tyre_model = LinearTyre()
    vehicle_kinematic = KinematicVehicleModel(dt=TIME_STEP)
    # vehicle_dynamic = DynamicVehicleModel()

    simulation = WayPointSimulation(sim_name="Training1", vehicle=vehicle_kinematic, input_data=data,
                                    time_step=TIME_STEP, timeout=60., iterations_per_step=50, way_point_threshold=0.5,
                                    distance_between_points=5.)
    observation_space = 1
    action_space = 5
    dqn = DQNController(observation_space=observation_space, action_space=action_space, check_name=SAVE_NAME)
    # dqn.model.load_weights("models/Working-2-Input-5-Output.hdf5")
    run = 0

    max_steps = 0
    max_step_run = 0
    most_points = 0
    most_points_run = 0
    has_solved = False
    solved_run = None

    avg_steps = 0
    avg_points_reached = 0

    try:
        while True:
            run += 1
            state = simulation.reset(dqn.exploration_rate)
            state = np.array([state[2]])  # Keep only theta1
            state = np.reshape(state, [1, observation_space])
            print(f"Initial state: {state}")
            step = 0
            total_reward = 0
            start = time.time()

            train_time = 0
            step_time = 0

            while True:

                step += 1
                simulation.render()
                action = dqn.act(state)
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
                state_next = np.array([state_next[2]])  # Keep only theta1
                state_next = np.reshape(state_next, [1, observation_space])
                dqn.remember(state=state, action=action, reward=reward, next_state=state_next, done=terminal)
                state = state_next

                if terminal:
                    avg_steps += step
                    avg_points_reached += points_reached

                    print("\n===================================================")
                    print("PREVIOUS RUN DATA")
                    print("===================================================")
                    print(f"Run: {run}, exploration: {dqn.exploration_rate}, steps: {step}")
                    print(f"Total sim time : {run_time}, Total points reached: {points_reached}")
                    print(f"Run end condition: {end_condition}")
                    print(f"Total rewards: {total_reward}")
                    print("===================================================\n")

                    if step >= max_steps:
                        max_steps = step
                        max_step_run = run
                    if points_reached >= most_points:
                        most_points = points_reached
                        most_points_run = run
                    if points_reached == len(data):
                        has_solved = True
                        solved_run = run

                    print("CUMULATIVE STATS")
                    print(f"Average steps: {avg_steps / run}")
                    print(f"Average points reached: {avg_points_reached / run}")
                    print(f"Longest run: {max_step_run} with {max_steps} steps")
                    print(f"Run with most points reached: {most_points_run} with {most_points} points")

                    # score_logger.add_score(step, run)

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

            if run == 2000:
                print(f"Run number {run} has been reached.")
                dqn.model.save("./DQNController1.hd5")
                print(f"Max score {max_steps} achieved in run {max_step_run}.")
                if has_solved:
                    print(f"DQN solved path in run {solved_run}.")
                else:
                    print("DQN failed to solve run.")
                exit(1)

            if run == 10:
                simulation.log_data(file_name="TestLog")

    except KeyboardInterrupt:
        print("User exit")
        print(f"Exited on run: {run}")
        print("===========================================================================")
        print("CUMULATIVE STATS")
        print(f"Average steps: {avg_steps / run}")
        print(f"Average points reached: {avg_points_reached / run}")
        print(f"Longest run: {max_step_run} with {max_steps} steps")
        print(f"Run with most points reached: {most_points_run} with {most_points} points")
        print("===========================================================================")
