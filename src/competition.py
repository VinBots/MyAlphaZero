######################################################
#
# AlphaZero algorithm applied to Tic-Tac-Toe
# written by Vincent Manier (vmanier2020@gmail.com)
#
# Module with functions to organize competitions
# between different agents: plain vanilla MCTS,
# MCTS with network and network-only
#
######################################################


import random
from copy import copy

import torch
import numpy as np

import games  # Games
import mcts
import policy  # neural network
from oracles import roll_out


def play_mcts(agent, iterations):
    orac_params = {"nb_roll_out": 1}
    for _ in range(iterations):
        agent.explore(orac_params)
    next_pos, (_, _, p) = agent.next(temperature=0.01)
    return next_pos.game.last_move


def network_only(game_state, policy_path):
    """
    Returns the best move (highest probability) output by the network
    """
    pol = policy.Policy()
    pol.load_weights(policy_path)
    board = torch.tensor(game_state).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    v, prob = pol.forward_batch(board)
    prob_array = prob.detach().numpy().reshape((3, 3))
    prob_array = prob_array * (np.abs(game_state) != 1).astype(np.uint8)
    return np.unravel_index(np.argmax(prob_array, axis=None), prob_array.shape)


def net_player(game, **kwargs):
    policy_path = kwargs["params"]["policy_path"]
    return network_only(game.player * game.state, policy_path)


def mcts_player(game, **kwargs):
    turn = kwargs["turn"]
    benchmark_competition_settings = kwargs["settings"]
    mcts_random_moves = benchmark_competition_settings.mcts_random_moves
    mcts_iterations = benchmark_competition_settings.mcts_iterations

    if turn < mcts_random_moves * 2:
        next_move = random.choice(game.available_moves())
    else:
        agent2 = mcts.Node(game, oracle=roll_out)
        next_move = play_mcts(agent2, mcts_iterations)
    return next_move


def match_net_mcts(game_settings, benchmark_competition_settings, match_params):

    player1 = match_params["player1"]
    player2 = match_params["player2"]
    inverse_order = match_params["inverse_order"]

    scores = 0
    inv_score = 1
    if inverse_order:
        inv_score = -1
    new_game = games.ConnectN(game_settings)
    players = [player1, player2]
    if inverse_order:
        players.reverse()
    turn = 0

    while new_game.score is None:
        if turn % 2 == 0:
            active_player = players[0]
        else:
            active_player = players[1]
        params = {
            "turn": turn,
            "settings": benchmark_competition_settings,
            "params": active_player[1],
        }
        next_move = active_player[0](new_game, **params)
        new_game.move(next_move)
        turn += 1
    scores = inv_score * new_game.score + 1
    return scores
