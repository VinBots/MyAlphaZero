import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def loss_chart(i):

    data = pd.read_csv("nn_loss-exp51.csv")
    x1 = data["iter"]
    y1 = data["loss"]

    data = pd.read_csv("nn_loss-exp6.csv")
    x2 = data["iter"]
    y2 = data["loss"]

    data = pd.read_csv("nn_loss-exp7.csv")
    x3 = data["iter"]
    y3 = data["loss"]

    data = pd.read_csv("nn_loss-exp8.csv")
    x4 = data["iter"]
    y4 = data["loss"]

    data = pd.read_csv("nn_loss-exp9.csv")
    x5 = data["iter"]
    y5 = data["loss"]

    data = pd.read_csv("nn_loss-exp10.csv")
    x6 = data["iter"]
    y6 = data["loss"]

    plt.cla()
    plt.plot(x1, y1, label="alpha = 2")
    plt.plot(x2, y2, label="alpha = 1")
    plt.plot(x3, y3, label="alpha = 0.5")
    plt.plot(x4, y4, label="alpha = 0.25")
    plt.plot(x5, y5, label="Disabled")
    plt.plot(x6, y6, label="alpha = 0.25 [2nd]")

    plt.legend(loc="upper right")
    plt.tight_layout()


matplotlib.rcParams["interactive"] == True
matplotlib.use("TkAgg")
plt.style.use("fivethirtyeight")
plt.title("Loss Function per training cycle")
ani = FuncAnimation(plt.gcf(), loss_chart, interval=3000)
plt.tight_layout()
plt.show()
