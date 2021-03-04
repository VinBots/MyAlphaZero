# Python libraries

# 3rd party libraries
import matplotlib.pyplot as plt
import numpy as np

# Game-related libraries
import games_mod # Games
import policy_mod # neural network
from play_mod import Play
import training_mod
from game_utils import DotDict

# Game settings
game_settings = DotDict({"board_size": (3, 3), "N": 3})

# Self-play training settings
game_training_settings = DotDict(
    {
        "comp_interval": 100,
        "episods": 200,
        "explore_steps": 50,
        "temperature_sch": np.array(
            [[300, 0.3], [10000, 0.01]]
        ),  # [x,y] means "up to x episods, applies y temperature"
    }
)
# neural network settings
nn_training_settings = DotDict(
    {
        "load_policy": False,
        "ai_ckp": "",
        "lr": 0.01,
        "weight_decay": 1.0e-4,
    }
)

# play settings
play_settings = DotDict({"explore_steps": 20, "temperature": 0.01})


def main():

    game = games_mod.ConnectN(game_settings)
    policy = policy_mod.Policy()
    alpha_0 = training_mod.AlphaZeroTraining(
        game_settings, game_training_settings, nn_training_settings, policy
    )


if __name__ == "__main__":
    main()
