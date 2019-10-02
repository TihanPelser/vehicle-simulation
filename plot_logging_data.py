import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import sys

style.use('fivethirtyeight')

LOG_FILE = ""

def animate(i):
    graph_data = open(LOG_FILE, 'r').read()
    lines = graph_data.split('\n')
    runs_list = []
    steps_list = []
    points_list = []
    rewards_list = []
    for line in lines:
        if len(line) > 1:
            run, steps, points, reward = line.split('\t')
            runs_list.append(float(run))
            steps_list.append(float(steps))
            points_list.append(float(points))
            rewards_list.append(float(reward))
    ax1.clear()
    ax1.plot(runs_list, steps_list, label="Steps")
    ax1.plot(runs_list, points_list, label="Points Reached")
    ax1.plot(runs_list, rewards_list, label="Rewards Obtained")
    ax1.set_xlabel("Run")
    ax1.legend()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please specify a file name!")
        exit(1)
    LOG_FILE = sys.argv[1]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()