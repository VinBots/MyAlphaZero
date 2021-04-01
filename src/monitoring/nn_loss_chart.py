import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

def loss_chart(i):
    data = pd.read_csv('nn_loss.csv')
    x = data['iter']
    y1 = data['loss']
    y2 = data['value_loss']
    y3 = data['prob_loss']
    plt.cla()
    plt.plot(x, y1, label='Losses')
    plt.plot(x, y2, label='MSE values')
    plt.plot(x, y3, label='Cross-Entropy probabilities')
    plt.legend(loc='upper right')    
    plt.tight_layout()
    
matplotlib.rcParams['interactive'] == True
matplotlib.use('TkAgg')
plt.style.use('fivethirtyeight')
plt.title("Loss Function per training cycle")
ani = FuncAnimation(plt.gcf(), loss_chart, interval=3000)
plt.tight_layout()
plt.show()