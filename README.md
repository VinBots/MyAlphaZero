# MyAlphaZero


This repository includes a work-in-progress implementation of AlphAZero algorithm applied to TicTacToe.

Please refer to docs/ai_lessons for status update

**Future works:**

* Better modularization
* Look at the use of a profiler
* Adjust the v value to reward faster wins v = win/loss * f_penalty (number of turns played - minimum)
* Review symetries function (2 are not operational)
* Test the competition of network
* alternate player1 and player2


**Final Objective**:

* Implement the AlphaZero Algorithm as described in the original paper, incl.
 * Asynchronous MCTS
 * Parallel implementations