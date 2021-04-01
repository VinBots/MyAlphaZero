import mcts
import games_mod
import torch
import numpy as np


def execute_self_play(game_settings, explore_steps, policy, temp, dir_eps, dir_alpha, dirichlet_enabled = False):
    """
    Starts with an empty board and runs MCTS for every node traversed.
    Experiences are stored in a buffer for the neural network to be trained.
    Returns a list of experiences in a list [state, value, probability]
    """
    memory = []
    mytree = mcts.Node(games_mod.ConnectN(game_settings))

    while mytree.outcome is None:
        for _ in range(explore_steps):
            mytree.explore(policy, dir_eps, dir_alpha, dirichlet_enabled)
        current_player = mytree.game.player
        mytree, (state, _, p) = mytree.next(temperature=temp)
        memory.append([state * current_player, p, current_player])
        mytree.detach_mother()
        outcome = mytree.outcome

    return [(m[0], m[2] * outcome, m[1]) for m in memory]