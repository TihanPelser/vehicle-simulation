import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

DIST_BETWEEN_POINTS = 5
FILE_NAME = "sin.txt"

if __name__ == '__main__':

    xp = np.arange(0, 100, 0.5)
    yp = 10*np.sin(np.deg2rad(xp*3.6))

    data_x = xp

    f = interp1d(xp, yp, kind="cubic")
    data_y = f(data_x)

    # Linear length on the line
    distance = np.cumsum(np.sqrt(np.ediff1d(data_x, to_begin=0)**2 + np.ediff1d(data_y, to_begin=0)**2))
    print(f"Before norm: {distance}")
    num_points = round(distance[-1] / DIST_BETWEEN_POINTS)

    distance = distance/distance[-1]
    print(f"After norm: {distance}")

    fx, fy = interp1d(distance, data_x), interp1d(distance, data_y)

    alpha = np.linspace(0, 1, num_points)
    x_regular, y_regular = fx(alpha), fy(alpha)

    xy = zip(x_regular, y_regular)

    with open(FILE_NAME, "w+") as f:
        for point in xy:
            f.write(str(round(point[0], 2)) + "," + str(round(point[1], 2)) + "\n")

    plt.figure()
    # plt.plot(xp, yp, "r.", label="Points")
    plt.plot(data_x, data_y, 'b-', label="Interpolated")
    plt.plot(x_regular, y_regular, 'or')
    # plt.axis('equal')
    plt.legend()
    plt.show()