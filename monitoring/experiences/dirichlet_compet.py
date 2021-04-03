import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def competition(i):
    data = pd.read_csv("compet-exp51.csv")
    x1 = data["iter"]
    y1 = data["scores"]

    data = pd.read_csv("compet-exp6.csv")
    x2 = data["iter"]
    y2 = data["scores"]

    data = pd.read_csv("compet-exp7.csv")
    x3 = data["iter"]
    y3 = data["scores"]

    data = pd.read_csv("compet-exp8.csv")
    x4 = data["iter"]
    y4 = data["scores"]

    data = pd.read_csv("compet-exp9.csv")
    x5 = data["iter"]
    y5 = data["scores"]

    data = pd.read_csv("compet-exp10.csv")
    x6 = data["iter"]
    y6 = data["scores"]

    plt.cla()
    plt.plot(x1, y1, label="alpha = 2")
    plt.plot(x2, y2, label="alpha = 1")
    plt.plot(x3, y3, label="alpha = 0.5")
    plt.plot(x4, y4, label="alpha = 0.25")
    plt.plot(x5, y5, label="Disabled")
    plt.plot(x6, y6, label="alpha = 0.25")

    plt.legend(loc="upper left")
    plt.tight_layout()


if __name__ == "__main__":
    matplotlib.rcParams["interactive"] == True
    matplotlib.use("TkAgg")
    plt.style.use("fivethirtyeight")
    plt.title("Competition Scores")
    ani = FuncAnimation(plt.gcf(), competition, interval=10000)
    plt.tight_layout()
    plt.show()
