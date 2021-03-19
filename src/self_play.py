import mcts
import games_mod
import torch
import numpy as np


def execute_self_play(game_settings, explore_steps, policy, temp):
    """
    Starts with an empty board and runs MCTS for every node traversed.
    Experiences are stored for the neural network to be trained.
    Returns a list of experiences
    """
    memory = []
    mytree = mcts.Node(games_mod.ConnectN(game_settings))

    while mytree.outcome is None:
        for _ in range(explore_steps):
            mytree.explore(policy)
        current_player = mytree.game.player
        # print ("The current player is {}, game state is {}".format(
        #    current_player, mytree.game.state))
        mytree, (state, _, p) = mytree.next(temperature=temp)

        # print ("Move is {}".format(mytree.game.state))

        memory.append([state * current_player, p, current_player])
        # print ("Sent to memory {}".format([state * current_player, p]))
        mytree.detach_mother()
        outcome = mytree.outcome

    result = [(m[0], m[2] * outcome, m[1]) for m in memory]
    # print ("sent to buffer {}".format (result))
    return [(m[0], m[2] * outcome, m[1]) for m in memory]


def fake_play():

    game_state1 = np.array([[-1, 1, -1], [0, 1, 0], [0, 0, 0]])
    outcome1 = 1
    p1 = torch.tensor(
        np.array([0.0000, 0.0000, 0.0000, 0.0, 0.0000, 0.5, 0.5, 0.0, 0.0])
    ).type(torch.FloatTensor)

    game_state2 = np.array([[1, 1, -1], [0, 1, -1], [0, 0, 0]])
    outcome2 = -1
    p2 = torch.tensor(
        np.array([0.0000, 0.0000, 0.0000, 0.0, 0.0000, 0.0, 0.0, 0.0, 1.0])
    ).type(torch.FloatTensor)

    return [(game_state1, outcome1, p1), (game_state2, outcome2, p2)]
