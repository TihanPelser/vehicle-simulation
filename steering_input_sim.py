from Simulation.Simulation import Simulation
from VehicleModel.Vehicle import Vehicle
from TyreModels.LinearCutoffTyreModel import LinearTyre
import numpy as np
import matplotlib.pyplot as plt

TRAINING_FILES = ["Sine_Steer.txt"]
TIMESTEP = 0.05


def read_data(file_name:str):
    file_data = []
    with open(f"SampleInputData/{file_name}", "r") as file:
        for line in file:
            vals = line.split(',')
            file_data.append([float(vals[0]), float(vals[1])])
    return file_data


if __name__ == "__main__":
    data = read_data(TRAINING_FILES[0])

    data = np.array(data[0::2])  # Take every second value

    tyre_model = LinearTyre()
    vehicle = Vehicle(tyre_model=tyre_model, dt=TIMESTEP)
    simulation = Simulation(sim_name="Training1", vehicle=vehicle, input_type="path", input_data=data,
                            timestep=TIMESTEP, timeout=50., waypoint_threshold=0.5)

    vehicle_x = []
    vehicle_y = []
    vehicle_theta = []

    for steering in data:
        print(f"Steering angle : {steering[1]} degrees")
        stats = vehicle.drive(steering_angle=np.radians(steering[1]))
        vehicle_x.append(stats[0])
        vehicle_y.append(stats[1])
        vehicle_theta.append(stats[2])

    params = vehicle.parameter_history()

    fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    plt.subplots_adjust(hspace = 0.5)

    ax1.plot(params["Time"], params["Delta"], label="Vehicle Steering Angle")
    ax1.plot(params["Time"], params["Theta"], label="Vehicle Heading")
    ax1.set_title("Applied Steering vs Vehicle Heading")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Angle [rad]")
    ax1.legend()

    ax2.plot(params["x"], params["y"], label="Vehicle Position")
    ax2.set_title("Vehicle Position")
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.legend()

    plt.show()

    fig = plt.figure()
    plt.plot(params["Time"], params["Force Front"], label="Front Tyre Force [N]")
    plt.plot(params["Time"], params["Force Rear"], label="Rear Tyre Force [N]")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()
    plt.title("Tyre Force Generation for Sine Steering Input")
    plt.legend()
    plt.show()