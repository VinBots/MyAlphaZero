#!/usr/bin/env python

######################################################
#
# AlphaZero algorithm applied to Tic-Tac-Toe
# written by Vincent Manier (vmanier2020@gmail.com)
# inspired by a case study provided by Udacity
#
######################################################

import time
import config as config
import policy_mod
import time
import games # Games
import training #neural network training
from replay_buffer_dict import ReplayBuffer #centralized buffer
from log_data import LogData #logging class for monitoring purposes
import cProfile
from utils import DotDict

def main():
    
    # Initialization of the logging files
    log_data = LogData()
    log_data.add_chart("nn_loss", ["nn_loss.csv", ['iter', 'loss', 'value_loss', 'prob_loss']])
    log_data.add_chart("buffer", ["buffer.csv", ['iter', 'wins', 'losses', 'draws']])
    log_data.add_chart("compet", ["compet.csv",['iter', 'scores']])
    log_data.add_chart("bias_action", ["bias_action.csv",['iter', 'pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'pos6', 'pos7', 'pos8', 'pos9']])

    # Initialization of the policy
    policy = policy_mod.Policy(config, log_data)

    buffer = ReplayBuffer(config.nn_training_settings.buffer_size_target, 
                        config.nn_training_settings.batch_size, 
                        log_data)

    t0 = time.time()
    alpha_0 = training.AlphaZeroTraining(
        config,
        policy,
        log_data)
    alpha_0.training_pipeline(buffer)
    t1 = time.time()

    print ("Total time (in sec): {}".format(t1 - t0))

if __name__ == "__main__":
    main()