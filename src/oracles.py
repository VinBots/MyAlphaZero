import random
from copy import copy

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def roll_out(game, **kwargs):
    nb_roll_out = kwargs["nb_roll_out"]
    scores = []
    
    for _ in range(nb_roll_out):
        sim_game = copy(game)
        while sim_game.score == None:
            random_move = random.choice(sim_game.available_moves())
            sim_game.move(random_move)
        scores.append(sim_game.score)
    v = game.player * np.average(np.array(scores))
    p = None
    return v, p

def nn_infer(game, **kwargs):
    policy = kwargs["policy"]
    input = (
        torch.tensor(
        game.state * game.player,
        dtype=torch.float,
        device=device
        )
        .unsqueeze(0).unsqueeze(0)
        )
    v, p = policy.forward_batch(input, dim_value=0)
    v = float(v.squeeze().squeeze())
    return v, p
