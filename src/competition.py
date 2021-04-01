import random
from copy import copy


import torch
import numpy as np

import games_mod  # Games
import plain_mcts
import mcts
import policy_mod  # neural network


def play_mcts(agent, iterations):
    for _ in range(iterations):
        agent.explore()
    agent = agent.next(temperature=0.01)
    return agent.game.last_move


def network_only(game_state, play_settings=None, policy_path="ckp/ai_ckp.pth"):
    """
    Returns the best move (highest probability) output by the network
    """

    policy = policy_mod.Policy(policy_path)
    policy.load_weights(policy_path)
    board = torch.tensor(game_state).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    v, prob = policy.forward_batch(board)
    prob_array = prob.detach().numpy().reshape((3, 3))
    prob_array = prob_array * (np.abs(game_state) != 1).astype(np.uint8)
    return np.unravel_index(np.argmax(prob_array, axis=None), prob_array.shape)


def match_net_mcts(
    policy, game_settings, benchmark_rounds, mcts_iterations, mcts_random_moves
):

    scores = 0
    draw_score = benchmark_rounds * 1
    for _ in range(benchmark_rounds):
        new_game = games_mod.ConnectN(game_settings)
        turn = 0
        while new_game.score is None:
            if turn % 2 == 0:
                next_move = network_only(new_game.state)
                new_game.move(next_move)
            else:
                if turn < mcts_random_moves * 2:
                    new_game.move(random.choice(new_game.available_moves()))
                else:
                    agent2 = plain_mcts.Node(new_game)
                    new_game.move(play_mcts(agent2, mcts_iterations))
            turn += 1

        scores += new_game.score + 1
        # print (turn, new_game.score)
        del new_game

    return scores / draw_score


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


"""
def match_mcts_net(policy, game_settings):
    
    num_matches = 50
    scores = 0
    for _ in range(num_matches):

        new_game = games_mod.ConnectN(game_settings)

        turn = 0
        #seq_states = []

        while new_game.score is None:
            #print (new_game.state)

            if turn % 2 == 1:
                next_move = network_only(-new_game.state)
                new_game.move(next_move)
            else:
                if turn >=10 :
                    new_game.move(random.choice(new_game.available_moves()))
                else:
                    agent2 = plain_mcts.Node(new_game)
                    new_game.move (play_mcts(agent2, 1000))
            turn +=1

        scores += new_game.score - 1
        #print (turn, new_game.score)
            
        del new_game
    
    #print (scores)
    return -scores
"""
