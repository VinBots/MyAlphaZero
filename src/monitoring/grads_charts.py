import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import argparse


def grads_chart(i, layer_name, indicator):
    """
    TO DO
    """

    data = pd.read_csv("grads.csv")
    x = data["iter"]

    fig = plt.figure(layer_name)
    plt.cla()
    y1 = data[layer_name + "_ave"]
    y2 = data[layer_name + "_max"]
    y3 = data[layer_name + "_min"]

    if indicator == "all" or indicator == "ave":
        y1 = data[layer_name + "_ave"]
        plt.plot(x, y1, label="Average")
    if indicator == "all" or indicator == "max":
        y2 = data[layer_name + "_max"]
        plt.plot(x, y2, label="Maximum")
    if indicator == "all" or indicator == "min":
        y3 = data[layer_name + "_min"]
        plt.plot(x, y3, label="Minimum")

    plt.title(layer_name)
    plt.legend(loc="upper right")
    plt.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose min, max, average or all")
    parser.add_argument("indicator", help="min for minimum...")

    args = parser.parse_args()
    indicator = args.indicator

    layers = [
        "conv.weight",
        "fc1.weight",
        "fc2.weight",
        "fc_action1.weight",
        "fc_action2.weight",
        "fc_value1.weight",
        "fc_value2.weight",
    ]

    ani = [0] * len(layers)
    matplotlib.rcParams["interactive"] == True
    matplotlib.use("TkAgg")
    plt.style.use("fivethirtyeight")
    for i in range(len(layers)):
        ani[i] = FuncAnimation(
            plt.figure(layers[i]),
            grads_chart,
            fargs=(layers[i], indicator),
            interval=10000,
        )
    plt.show()
