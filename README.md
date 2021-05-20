
# MyAlphaZero

This repository includes an implementation of AlphaZero algorithm applied to the TicTacToe game, based on Deepmind's paper, Mastering the game of Go without human knowledge, published in October 2017.

# Learning Objectives
* Self-play reinforcement learning techniques
* Monte Carlo Tree Search (MCTS)
* Implementation details (e.g. dirichlet noise, masking)
* Parallel implementations (via multiple processes) of game generations
* Python programming techniques

# Credits

* Jonathan Laurent (https://github.com/jonathan-laurent) for his supervision and guidance
* Udacity Deep Reinforcement Learning Nanodegree (https://github.com/udacity/deep-reinforcement-learning)

# References

* Silver, D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017)

# Installation

1. Clone the repository

```
git clone https://github.com/VinBots/MyAlphaZero.git
```

2. Create and activate a new virtual environment via conda

```
conda create --name new_env python=3.6.13
conda activate new_env
```

3. Install the required packages

Go to the root directory and install the dependencies
```
cd MyAlphaZero
pip install -r requirements.txt
```

4. Run the algorithm

```
python main.py
```