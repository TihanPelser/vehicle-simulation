import numpy as np
from scipy import interpolate
import time
from TyreModel import TyreModel

import pandas as pd
import yaml

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numba import jit

data = pd.read_csv("Tyre_Model.txt", sep="\t").values

slip = 2.5
n = 500

tyre = TyreModel()


x = np.linspace(0, np.radians(20), num=200)
y = np.linspace(data[:,2].min(), data[:,2].max(), num=200)

xx, yy = np.meshgrid(x, y)

xy = np.stack((np.radians(data[:, 0]),data[:, 2]), axis=1)

print(xy.shape)

xi = np.stack((xx.flatten(), yy.flatten()), axis=1)

run = time.time()
interpolation = interpolate.CloughTocher2DInterpolator(points=xy, values=data[:, 1])
print("CloughTocher fit time: ", time.time() - run)
run = time.time()


g = interpolate.griddata(xy, values=data[:, 1], xi=xi, method="cubic")
print("Griddata interp time: ", time.time() - run)
run = time.time()

g = interpolation(xi)
print("CloughTocher interp time: ", time.time() - run)
run = time.time()

# g = interpolate.griddata(xy, values=data[:, 1], xi=xi, method="linear")
print("Linear interp time: ", time.time() - run)

g = g.reshape((200, 200))

fig = plt.figure()
ax = fig.gca(projection='3d')
# plt.set_cmap('jet')
plot = ax.plot_surface(xx, yy, g, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_title("Tyre Model Surface")
ax.set_zlabel("Lateral Force [N]")
# fig.colorbar(plot, shrink=0.5, aspect=5)
plt.xlabel("Slip Angle [Rad]")
plt.ylabel("Normal Force [N]")

fig.colorbar(plot, shrink=0.5, aspect=5)
# plot.zlabel("Lateral Force [N]")
plt.show()

# fig = plt.figure()
#
# ax = fig.gca(projection='3d')
# # Plot the surface.
# surf = ax.plot_surface(xx, yy, g, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#
# # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()
