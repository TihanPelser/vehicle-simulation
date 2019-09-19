import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv(f"Tyre_Model.txt", sep="\t").values


# slip = data[:, 0]
# fy = data[:, 1]
# normal = data[:, 2]

load1_angle = []
load2_angle = []
load3_angle = []

load1_force = []
load2_force = []
load3_force = []

x1 = []
x2 = []
x3 = []
y1 = []
y2 = []
y3 = []

for i in data:
    if i[2] == 402.21:
        load1_angle.append(i[0])
        load1_force.append(i[1])
        if i[0] <= 10.0:
            x1.append(i[0])
            y1.append(i[1])
    elif i[2] == 608.22:
        load2_angle.append(i[0])
        load2_force.append(i[1])
        if i[0] <= 10.0:
            x2.append(i[0])
            y2.append(i[1])
    elif i[2] == 809.325:
        load3_angle.append(i[0])
        load3_force.append(i[1])
        if i[0] <= 10.0:
            x3.append(i[0])
            y3.append(i[1])

x1 = np.radians(np.array(x1))
x2 = np.radians(np.array(x2))
x3 = np.radians(np.array(x3))
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)

a1, _, _, _ = np.linalg.lstsq(x1[:, np.newaxis], y1)
a2, _, _, _ = np.linalg.lstsq(x2[:, np.newaxis], y2)
a3, _, _, _ = np.linalg.lstsq(x3[:, np.newaxis], y3)

print(a1, a2, a3)

line1_x = [0., x1.max(), np.radians(20.)]
line1_y = [0., a1*x1.max(), a1*x1.max()]

line2_x = [0., x2.max(), np.radians(20.)]
line2_y = [0., a2*x2.max(), a2*x2.max()]

line3_x = [0., x3.max(), np.radians(20.)]
line3_y = [0., a3*x3.max(), a3*x3.max()]

# Sort load lists
load1_angle, load1_force = zip(*sorted(zip(load1_angle, load1_force)))
load2_angle, load2_force = zip(*sorted(zip(load2_angle, load2_force)))
load3_angle, load3_force = zip(*sorted(zip(load3_angle, load3_force)))

        
fig = plt.figure(figsize=(10, 8), dpi=100)
plt.scatter(np.radians(load1_angle), load1_force, label=f"{round(41*9.81,2)}N Load Measured Data", color="b")
plt.scatter(np.radians(load2_angle), load2_force, label=f"{round(62*9.81,2)}N Load Measured Data", color="orange")
plt.scatter(np.radians(load3_angle), load3_force, label=f"{round(82.5*9.81,2)}N Load Measured Data", color="g")
plt.plot(np.radians(load1_angle), load1_force, color="b", linestyle='-', linewidth=0.3)
plt.plot(np.radians(load2_angle), load2_force, color="orange", linestyle='-', linewidth=0.3)
plt.plot(np.radians(load3_angle), load3_force,  color="g", linestyle='-', linewidth=0.3)


plt.title("Slip Angle vs Lateral Force for Various Load Values")
plt.legend(loc=2)
plt.xlabel("Slip Angle [rad]")
plt.ylabel("Lateral Force [N]")
plt.xlim(0., np.radians(20.1))
plt.ylim(0., )
plt.show()

fig = plt.figure(figsize=(10, 8), dpi=100)

plt.plot(line1_x, line1_y, linestyle='-', color='b', label=f"{round(41*9.81,2)}N Load Fitted Line")
plt.plot(line2_x, line2_y, linestyle='-', color='orange', label=f"{round(62*9.81, 2)}N Load Fitted Line")
plt.plot(line3_x, line3_y, linestyle='-', color='g', label=f"{round(82.5*9.81,2)}N Load Fitted Line")
plt.plot(np.radians([10., 10]), [-10, 1000], linestyle='--', color='red', linewidth=0.3)

plt.title("Linear Cutoff Model for Various Load Values")
plt.legend(loc=2)
plt.xlabel("Slip Angle [rad]")
plt.ylabel("Lateral Force [N]")
plt.xlim(0., np.radians(20.))
plt.ylim(0., 450.)
plt.show()