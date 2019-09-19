from TyreModels import TyreModelNN
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


tyre = TyreModelNN.NNTyreModel(config=None)

x = np.linspace(0, np.radians(20), num=200)
y = np.linspace(tyre.data[:, 2].min(), tyre.data[:, 2].max(), num=200)


xx, yy = np.meshgrid(x, y)

xi = np.stack((xx.flatten(), yy.flatten()), axis=1)

predicted = tyre.model.predict(xi)

print(predicted)

predicted = predicted.reshape((200, 200))

fig = plt.figure()
ax = fig.gca(projection='3d')
# plt.set_cmap('jet')
plot = ax.plot_surface(xx, yy, predicted, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_title("Tyre Model Surface")
ax.set_zlabel("Lateral Force [N]")
# fig.colorbar(plot, shrink=0.5, aspect=5)
plt.xlabel("Slip Angle [Rad]")
plt.ylabel("Normal Force [N]")

fig.colorbar(plot, shrink=0.5, aspect=5)
# plot.zlabel("Lateral Force [N]")
plt.show()