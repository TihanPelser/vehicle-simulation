import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Double Lane Change ISO Standard
# Defined Points x [0., 7.5, 15., 45., 57.5, 70., 95., 125.]
# Defined points y [0., 0., 0., 3.5, 3.5, 3.5, 0., 0.]

# x_1 = np.arange(0, 15.5, 0.5)
# y_1 = np.zeros(x_1.shape)
# gap1_x = np.arange(15., 45.5, 0.5)
# x_2 = np.arange(45., 70.5, 0.5)
# y_2 = np.ones(x_2.shape) * 3.5
# gap2_x = np.arange(70., 95.5, 0.5)
# x_3 = np.arange(95., 125.5, 0.5)
# y_3 = np.zeros(x_3.shape)
# 
# xp = np.concatenate((x_1, x_2, x_3), axis=0)
# yp = np.concatenate((y_1, y_2, y_3), axis=0)
# 
# data_x = np.arange(0., 125.5, 0.5)
# 
# f = interp1d(xp, yp, kind="cubic")
# data_y = f(data_x)

x = np.arange(0, 100.5, 0.5)
y = 20 * np.sin(np.radians(3.6*x))

with open("Sine_Steer.txt", "w+") as f:
    for i in range(len(x)):
        f.write(str(x[i]) + "," + str(y[i]) + "\n")

plt.figure()
# plt.plot(xp, yp, "r.", label="Points")
plt.plot(x, y, 'b-', label="Interpolated")
plt.legend()
plt.show()