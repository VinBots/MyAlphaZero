import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def competition(i):
    data = pd.read_csv("compet.csv")
    x = data["iter"]
    y1 = data["scores"]

    plt.cla()
    plt.axhline(y = 1.0, label = 'benchmark', color = 'r', linestyle = 'dashed')
    plt.plot(x, y1, label="Network vs. MCTS 1000", color = 'g')
    plt.legend(loc="lower right")
    plt.tight_layout()


if __name__ == "__main__":
    matplotlib.rcParams["interactive"] == True
    matplotlib.use("TkAgg")
    plt.style.use("fivethirtyeight")
    plt.title("Competition Scores")
    ani = FuncAnimation(plt.gcf(), competition, interval=3000)
    plt.tight_layout()
    plt.show()
