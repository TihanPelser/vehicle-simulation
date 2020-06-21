import matplotlib.pyplot as plt
import numpy as np
from vehicle_models.kinematic_model import KinematicVehicleModel

if __name__ == "__main__":
    TIME_STEP = 0.001
    TIME = np.linspace(0, 10, 100)
    STEER_INPUT = np.deg2rad(15) * np.sin(np.deg2rad(TIME * 36))
    print(STEER_INPUT.shape)
    X = []
    Y = []
    THETA = []
    vehicle_kinematic = KinematicVehicleModel(dt=TIME_STEP)
    for steer in STEER_INPUT:
        print(steer)
        x, y, theta = vehicle_kinematic.drive(steering_angle=steer)
        X.append(x)
        Y.append(Y)
        THETA.append(theta)

    print(X)
    print(Y)
    print(THETA)
    fig, (ax1, ax2) = plt.subplots(2)
    plt.subplots_adjust(hspace=0.6)
    ax1.plot(TIME, STEER_INPUT, label="Steering Input")
    ax1.plot(TIME, THETA, label="Vehicle Heading")
    ax1.set_title("Steering Input vs Vehicle Heading")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Angle [rad]")
    # ax1.set_xlim((0, 60))
    # ax1.set_ylim((-10, 10))
    ax1.grid()
    ax1.legend()

    ax2.set_title("Vehicle Position")
    ax2.plot(X, Y)

    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    # ax2.set_xlim((0, 22.5))
    ax2.grid()
    # plt.savefig("plots/actual_vs_simulated_results.png")
    plt.show()
