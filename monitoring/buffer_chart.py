import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def buffer_chart(i):
    data = pd.read_csv("buffer.csv")
    x = data["iter"]
    y1 = data["wins"]
    y2 = data["losses"]
    y3 = data["draws"]
    plt.cla()
    plt.plot(x, y1, label="wins")
    plt.plot(x, y2, label="losses")
    plt.plot(x, y3, label="draws")
    plt.legend(loc="upper right")
    plt.tight_layout()


if __name__ == "__main__":

    matplotlib.rcParams["interactive"] == True
    matplotlib.use("TkAgg")
    plt.style.use("fivethirtyeight")
    plt.title("Distributions of wins, losses and draws")
    ani = FuncAnimation(plt.gcf(), buffer_chart, interval=3000)
    plt.tight_layout()
    plt.show()
