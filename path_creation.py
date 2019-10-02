import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

distance_between_points = 5

# Double Lane Change ISO Standard
# Defined Points x [0., 7.5, 15., 45., 57.5, 70., 95., 125.]
# Defined points y [0., 0., 0., 3.5, 3.5, 3.5, 0., 0.]

x_1 = np.arange(0, 15.5, 0.5)
y_1 = np.zeros(x_1.shape)
gap1_x = np.arange(15., 45.5, 0.5)
x_2 = np.arange(45., 70.5, 0.5)
y_2 = np.ones(x_2.shape) * 3.5
gap2_x = np.arange(70., 95.5, 0.5)
x_3 = np.arange(95., 125.5, 0.5)
y_3 = np.zeros(x_3.shape)

xp = np.concatenate((x_1, x_2, x_3), axis=0)
yp = np.concatenate((y_1, y_2, y_3), axis=0)

data_x = np.arange(0., 125.1, 0.1)

f = interp1d(xp, yp, kind="cubic")
data_y = f(data_x)

# Linear length on the line
distance = np.cumsum(np.sqrt(np.ediff1d(data_x, to_begin=0)**2 + np.ediff1d(data_y, to_begin=0)**2))
print(f"Before norm: {distance}")
num_points = round(distance[-1]/distance_between_points)

distance = distance/distance[-1]
print(f"After norm: {distance}")

fx, fy = interp1d(distance, data_x), interp1d(distance, data_y)


alpha = np.linspace(0, 1, num_points)
x_regular, y_regular = fx(alpha), fy(alpha)

# x = np.arange(0, 100.5, 0.5)
# y = 20 * np.sin(np.radians(3.6*x))
xy = zip(x_regular, y_regular)

with open("DLC.txt", "w+") as f:
    for point in xy:
        f.write(str(round(point[0],2)) + "," + str(round(point[1],2)) + "\n")

plt.figure()
# plt.plot(xp, yp, "r.", label="Points")
plt.plot(data_x, data_y, 'b-', label="Interpolated")
plt.plot(x_regular, y_regular, 'or')
# plt.axis('equal')
plt.legend()
plt.show()