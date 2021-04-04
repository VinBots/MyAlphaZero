import random
from copy import copy

import torch
import numpy as np

import games_mod  # Games
import mcts
import policy_mod  # neural network
from oracles import roll_out


def play_mcts(agent, iterations):
    orac_params = {"nb_roll_out":1}
    for _ in range(iterations):
        agent.explore(orac_params)
    next_pos, (_, _, p) = agent.next(temperature=0.01)
    #print ("Probs = {}".format(p))
    return next_pos.game.last_move

def network_only(game_state, policy_path, nn_training_settings):
    """
    Returns the best move (highest probability) output by the network
    """
    policy = policy_mod.Policy(policy_path, nn_training_settings)
    #print (policy_path)
    policy.load_weights(policy_path)
    board = torch.tensor(game_state).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    v, prob = policy.forward_batch(board)
    prob_array = prob.detach().numpy().reshape((3, 3))
    prob_array = prob_array * (np.abs(game_state) != 1).astype(np.uint8)
    return np.unravel_index(np.argmax(prob_array, axis=None), prob_array.shape)

def net_player (game, **kwargs):
    #print ("Net is playing {}".format(game.player))
    policy_path = kwargs["params"]["policy_path"]
    nn_training_settings = kwargs["params"]["nn_training_settings"]
    return network_only(game.player * game.state, policy_path, nn_training_settings)

def mcts_player(game, **kwargs):
    turn = kwargs["turn"]
    benchmark_competition_settings = kwargs["settings"]
    mcts_random_moves = benchmark_competition_settings.mcts_random_moves
    mcts_iterations = benchmark_competition_settings.mcts_iterations

    if turn < mcts_random_moves * 2:
        next_move = random.choice(game.available_moves())
    else:
        agent2 = mcts.Node(game, oracle = roll_out)
        next_move = play_mcts(agent2, mcts_iterations)
    return next_move

def match_net_mcts(game_settings, benchmark_competition_settings, **kwargs):

    player1 = kwargs["player1"]
    player2 = kwargs["player2"]
    inverse_order = kwargs["inverse_order"]

    scores = 0
    inv_score = 1
    if inverse_order:
        inv_score = -1
    new_game = games_mod.ConnectN(game_settings)
    players = [player1, player2]
    if inverse_order:
        players.reverse()
    turn = 0

    while new_game.score is None:
        if turn % 2 == 0:
            active_player = players[0]
        else:
            active_player = players[1]
        params = {"turn": turn, "settings":benchmark_competition_settings, "params": active_player[1]}        
        next_move = active_player[0](new_game, **params)
        new_game.move(next_move)
        turn += 1
    scores = inv_score * new_game.score + 1
    return scores

def policy_player_mcts(game, play_settings=None, policy_path="ckp/ai_ckp.pth"):
    """to do"""

    explore_steps = play_settings.explore_steps
    temperature = play_settings.temperature

    policy = policy_mod.Policy("")
    policy.load_weights(policy_path)

    mytree = mcts.Node(copy(game))

    for _ in range(explore_steps):
        mytree.explore(policy, _, _, False)

    mytreenext, (_, _, _) = mytree.next(temperature=temperature)
    return mytreenext.game.last_move


def match_ai(game_settings, play_settings, player1_func, player2_func, total_rounds=1):
    """TO DO"""

    scores = 0
    player_turn = 1
    path = "ckp/ai_temp_ckp.pth"

    for _ in range(total_rounds):
        player_turn = 1
        game = games_mod.ConnectN(game_settings)
        player1 = player1_func
        player2 = player2_func
        curr_player = player1
        score = None

        while score is None:
            # print (game.state)
            game_state = player_turn * game.state
            loc = curr_player(game, play_settings, path)

            succeed = game.move(loc)
            if succeed:
                score = game.score
                if player_turn == 1:
                    curr_player = player2
                    player_turn = -1
                    path = "ckp/ai_ckp.pth"
                else:
                    curr_player = player1
                    player_turn = 1
                    path = "ckp/ai_temp_ckp.pth"

        scores += score

    return scores