import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def bias(i):
    data = pd.read_csv("bias_action.csv")
    x = data["iter"]
    y_list = [data["pos"+str(i)] for i in range(1, 10)]

    plt.cla()
    #plt.plot(x, y_list[4], label="pos" + str(i))

    for i in range(9):
        plt.plot(x, y_list[i], label="pos" + str(i))
    
    plt.legend(loc="lower right")
    plt.tight_layout()


if __name__ == "__main__":
    matplotlib.rcParams["interactive"] == True
    matplotlib.use("TkAgg")
    plt.style.use("classic")

    #plt.style.use("fivethirtyeight")
    plt.title("Bias Values")
    ani = FuncAnimation(plt.gcf(), bias, interval=3000)
    plt.tight_layout()
    plt.show()
