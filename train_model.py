from Controller.DQNController import DQNController
from Controller.Scoring import ScoreLogger
from Simulation.Simulation import Simulation
from VehicleModel.Vehicle import Vehicle
from TyreModels.LinearCutoffTyreModel import LinearTyre
import numpy as np
import time


TRAINING_FILES = ["DoubleLaneChange.txt", "Sine.txt"]
TIMESTEP = 0.01
SAVE_NAME = "LOG_SCORE_DQN_HUBER_LOSS"


def read_data(file_name: str):
    file_data = []
    with open(f"TrainingData/{file_name}", "r") as file:
        for line in file:
            vals = line.split(',')
            file_data.append([float(vals[0]), float(vals[1])])
    return file_data


if __name__ == "__main__":

    data = read_data(TRAINING_FILES[0])

    data = np.array(data[0::3])  # Take every third value

    tyre_model = LinearTyre()
    vehicle = Vehicle(tyre_model=tyre_model, dt=TIMESTEP)
    simulation = Simulation(sim_name="Training1", vehicle=vehicle, input_type="path", input_data=data,
                            timestep=TIMESTEP, timeout=50., waypoint_threshold=0.5)
    score_logger = ScoreLogger("Training1")
    observation_space = 4
    action_space = 2
    dqn = DQNController(observation_space, action_space, check_name=SAVE_NAME)
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
            state = simulation.reset()
            # print(state)
            state = np.reshape(state, [1, observation_space])
            step = 0
            total_reward = 0
            start = time.time()

            train_time = 0
            step_time = 0

            while True:

                step += 1
                # simulation.render()
                action = dqn.act(state)
                # state_next, reward, points_reached, terminal, time = simulation.step(step_type="action", input=action)
                step_time_taken = time.time()
                results = simulation.step(step_type="action", input=action)
                step_time += time.time() - step_time_taken

                state_next = results.state
                reward = results.reward
                terminal = results.terminal
                points_reached = results.progress
                run_time = results.run_time
                end_condition = results.end_condition

                # reward = reward if not terminal else -reward

                total_reward += reward

                state_next = np.reshape(state_next, [1, observation_space])
                dqn.remember(state, action, reward, state_next, terminal)
                state = state_next

                if terminal:
                    avg_steps += step
                    avg_points_reached += points_reached

                    print("=============================================================")
                    print(f"Run: {run}, exploration: {dqn.exploration_rate}, steps: {step}")
                    print(f"Total sim time : {run_time}, Total points reached: {points_reached}")
                    print(f"Run end condition: {end_condition}")
                    print(f"Total rewards: {total_reward}")
                    print("=============================================================")

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
