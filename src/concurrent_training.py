import threading
import multiprocessing
import config
import time
#import numpy as np
#import random
#import torch
#from tqdm import trange

import mcts
import games_mod
#from log_data import LogData
import policy_mod
from oracles import roll_out, nn_infer


def execute_self_play(policy):
    """
    """

    game_settings = config.game_settings
    mcts_settings = config.mcts_settings
    
    memory = []
    mytree = mcts.Node(games_mod.ConnectN(game_settings), oracle = nn_infer)
    orac_params = {"policy":policy}

    while mytree.outcome is None:
        for _ in range(mcts_settings.explore_steps):
            mytree.explore(
                orac_params, 
                dir_eps = mcts_settings.dir_eps, 
                dir_alpha =  mcts_settings.dir_alpha,
                dirichlet_enabled= mcts_settings.dir_enabled)
        current_player = mytree.game.player
        mytree, (state, _, p) = mytree.next(temperature=mcts_settings.temp)
        memory.append([state * current_player, p, current_player])
        mytree.detach_mother()
        outcome = mytree.outcome

    return [(m[0], m[2] * outcome, m[1]) for m in memory]


def concurrent_training():
    policy = policy_mod.Policy(config.nn_training_settings.policy_path, 
                            config.nn_training_settings)
    policy.load_weights(config.nn_training_settings.policy_path)

    self_play_iterations = 50
    start = time.perf_counter()
    threads = []
    for _ in range(self_play_iterations):
        # BASE CASE
        '''
        execute_self_play(policy)
        '''
        #EXPERIENCE 1
        '''
        t = threading.Thread (target = execute_self_play)
        t.start()
        threads.append(t)
    #for t in threads:
        t.join()
        '''
        #EXPERIENCE 2
        
        t = multiprocessing.Process (target = execute_self_play, args = (policy,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
        
    
    end = time.perf_counter()
    print ("Total time per iteration: {}".format((end - start)/self_play_iterations))

if __name__ == "__main__":
    concurrent_training()
    
