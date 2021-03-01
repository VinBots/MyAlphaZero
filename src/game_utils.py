from copy import copy
import random

import torch.optim as optim
import torch

import policy_mod
import mcts
import games_mod


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


def policy_player_mcts(game, policy_path="ai_ckp.pth"):
    """to do"""

    policy = policy_mod.Policy()
    policy.load_weights(policy_path)
    mytree = mcts.Node(copy(game))
    for _ in range(50):
        mytree.explore(policy)
    mytreenext, (v, nn_v, p, nn_p) = mytree.next(temperature=0.1)
    return mytreenext.game.last_move


def random_player(game):
    return random.choice(game.available_moves())


