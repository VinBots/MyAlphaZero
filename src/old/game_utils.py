from copy import copy
import random

import numpy as np
import torch.optim as optim
import torch

import policy_mod
import mcts
import games_mod


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


def policy_player_mcts(game, play_settings=None, policy_path="ai_ckp.pth"):
    """to do"""

    explore_steps = play_settings.explore_steps
    temperature = play_settings.temperature

    policy = policy_mod.Policy()
    policy.load_weights(policy_path)
    mytree = mcts.Node(copy(game))
    for _ in range(explore_steps):
        mytree.explore(policy)
    mytreenext, (
        _,
        _,
        _,
    ) = mytree.next(temperature=temperature)
    return mytreenext.game.last_move


def network_only (game_state, play_settings=None, policy_path="ai_ckp.pth"):
    """to do"""
    #print ("NETWORK ONLY PLAYING")

    policy = policy_mod.Policy()
    policy.load_weights(policy_path)
    board = torch.tensor(game_state).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)

    v, prob = policy.forward_batch(board)
    prob_array = prob.detach().numpy().reshape((3,3))
    prob_array = prob_array * (np.abs(game_state) != 1).astype(np.uint8)
    print (board, prob_array)
    return np.unravel_index(np.argmax(prob_array, axis=None), prob_array.shape)


def random_player(game, play_settings=None):
    return random.choice(game.available_moves())

def match_ai(game_settings, play_settings, player1_func, player2_func, total_rounds=1):
    """TO DO"""

    total_wins = 0
    total_losses = 0
    player_turn = 1
    
    for _ in range(total_rounds):
        player_turn = 1
        game = games_mod.ConnectN(game_settings)
        player1 = player1_func
        player2 = player2_func
        
        curr_player = player1
        score = None

        while score is None:
            print (game.state)
            game_state = player_turn * game.state
            loc = curr_player(game_state, play_settings)
            
            succeed = game.move(loc)

            if succeed:
                score = game.score
                if player_turn == 1:
                    curr_player = player2
                    player_turn = -1
                else:
                    curr_player = player1
                    player_turn = 1
        
        if score == 1:
            total_wins += score
        elif score == -1:
            total_losses += -score

    print("Total wins / losses of Player 1 : {} / {}".format(total_wins, total_losses))
    return (total_wins, total_losses)
