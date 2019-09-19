from Controller.DQNController import DQNController
from Controller.Scoring import ScoreLogger
from Simulation.Simulation import Simulation
from VehicleModel.Vehicle import Vehicle
from TyreModels.LinearCutoffTyreModel import LinearTyre
import numpy as np

TRAINING_FILES = ["DoubleLaneChange.txt", "Sine.txt"]
TIMESTEP = 0.001


def read_data(file_name:str):
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
    dqn = DQNController(observation_space, action_space)
    run = 0

    max_score = 0
    max_score_run = 0
    has_solved = False
    solved_run = None

    while True:
        run += 1
        state = simulation.reset()
        # print(state)
        state = np.reshape(state, [1, observation_space])
        step = 0
        total_reward =0
        while True:
            step += 1
            #env.render()
            action = dqn.act(state)
            state_next, reward, points_reached, terminal, time = simulation.step(step_type="action", input=action)

            reward = reward if not terminal else -reward

            total_reward += reward

            state_next = np.reshape(state_next, [1, observation_space])
            dqn.remember(state, action, reward, state_next, terminal)
            state = state_next

            if terminal:
                print(f"Run: {run}, exploration: {dqn.exploration_rate}, score: {step}")
                print(f"Total sim time : {time}, Total points reached: {points_reached}")
                if total_reward >= max_score:
                    max_score = total_reward
                    max_score_run = run
                if points_reached == len(data):
                    has_solved = True
                    solved_run = run

                # score_logger.add_score(step, run)
                break
            dqn.experience_replay()

            if run == 2000:
                print(f"Run number {run} has been reached.")
                dqn.model.save("./DQNController1.hd5")
                print(f"Max score {max_score} achieved in run {max_score_run}.")
                if has_solved:
                    print(f"DQN solved path in run {solved_run}.")
                else:
                    print("DQN failed to solve run.")
                exit(1)

            
