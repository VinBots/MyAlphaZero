

<h1 align="center">
  <br>
  <a href="https://github.com/VinBots/MyAlphaZero"><img src="docs/assets/logo_alpha_zero.jpg" alt="AlphaZero"></a>
</h1>

<h4 align="center">AlphaZero algorithm applied to Tic-Tac-Toe game</h4>
<p align="center">
  <a href="#about">About</a> •
  <a href="#learning">Learning</a> •
  <a href="#installation">Installation</a> •
  <a href="#references">References</a> •
  <a href="#credits">Credits</a>
</p>

---


## About

This repository includes an implementation of AlphaZero algorithm applied to the TicTacToe game, based on Deepmind's paper, Mastering the game of Go without human knowledge, published in October 2017.


## Learning
* Self-play reinforcement learning techniques
* Monte Carlo Tree Search (MCTS)
* AlphaZero algorithm implementation details (e.g. dirichlet noise, masking)
* Parallel implementations (via multiple processes) of game generations
* Python programming techniques

## Installation

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
1. Run the algorithm
```
python main.py
```

## References

* Silver, D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017)


## Credits

* Jonathan Laurent (https://github.com/jonathan-laurent) for his supervision and guidance
* Udacity Deep Reinforcement Learning Nanodegree (https://github.com/udacity/deep-reinforcement-learning)



