# MyAlphaZero


This repository includes a work-in-progress implementation of AlphAZero algorithm applied to TicTacToe.


**Future works:**

* Use native Pytorch loss function - DONE
* Remove masking in the forward pass and in the loss calculation - DONE
* Use a hashing table for the buffer - DONE total time divided by 4-5
* Dirichlet distribution - DONE
* Ensure diversification of experiences is not done at the expense of weakening MCTS - DONE
* Improve the deduplication functions (averages) DONE
* Stats about the network, buffer, performance
* Find a better way to update stats - DONE
* Organize competition against baseline



* Better modularization
* Look at the use of a profiler
* Adjust the v value to reward faster wins v = win/loss * f_penalty (number of turns played - minimum)
* Review symetries function (2 are not operational)


**Final Objective**:

* Implement the AlphaZero Algorithm as described in the original paper
 * Asynchronous MCTS
 * Parallel implementations

 
 
``
def animate(i):
    data = pd.read_csv('data.csv')
    x = data['x_value']
    y1 = data['total_1']
    y2 = data['total_2']

    ax = plt.gca()
    line1, line2 = ax.lines

    line1.set_data(x, y1)
    line2.set_data(x, y2)

    xlim_low, xlim_high = ax.get_xlim()
    ylim_low, ylim_high = ax.get_ylim()

    ax.set_xlim(xlim_low, (x.max() + 5))

    y1max = y1.max()
    y2max = y2.max()
    current_ymax = y1max if (y1max > y2max) else y2max

    y1min = y1.min()
    y2min = y2.min()
    current_ymin = y1min if (y1min < y2min) else y2min

    ax.set_ylim((current_ymin - 5), (current_ymax + 5)) 
``

