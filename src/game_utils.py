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

def policy_player_mcts(game, play_settings=None, policy_path="ai_ckp.pth"):
    """to do"""

    explore_steps = play_settings.explore_steps
    temperature = play_settings.temperature
        
    policy = policy_mod.Policy()
    policy.load_weights(policy_path)
    mytree = mcts.Node(copy(game))
    for _ in range(explore_steps):
        mytree.explore(policy)
    mytreenext, (_, _, _,) = mytree.next(temperature=temperature)
    return mytreenext.game.last_move


def policy_player_pure_mcts(game, play_settings=None):
    """to do"""

    explore_steps = play_settings.explore_steps
    temperature = play_settings.temperature
        
    policy = policy_mod.Policy()
    mytree = mcts.Node(copy(game))
    for _ in range(explore_steps):
        mytree.explore(policy)
    mytreenext, (_, _, _,) = mytree.next(temperature=temperature)
    return mytreenext.game.last_move


def random_player(game, play_settings=None):
    return random.choice(game.available_moves())

def match_ai(game_settings, play_settings, total_rounds=100):
    """TO DO"""

    total_wins = 0
    total_losses = 0

    for _ in range(total_rounds):
        game = games_mod.ConnectN(game_settings)
        player1 = policy_player_mcts
        player2 = policy_player_mcts
        curr_player = player1
        score = None

        while score is None:
            loc = curr_player(game, play_settings)
            succeed = game.move(loc)
            if succeed:
                score=game.score
                if curr_player == player1:
                    curr_player = player2
                else:
                    curr_player = player1
        if score == 1:
            total_wins+=score
        elif score == -1:
            total_losses+=-score

    print ("Total wins / losses of Player 1 : {} / {}".format(total_wins, total_losses))
    return (total_wins, total_losses)
