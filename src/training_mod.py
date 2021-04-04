import numpy as np
import random
import torch
from tqdm import trange

import mcts
import games_mod
from log_data import LogData
from competition import match_net_mcts, match_ai, policy_player_mcts, net_player, mcts_player
import policy_mod
from oracles import roll_out, nn_infer


def execute_self_play(
    game_settings,
    mcts_settings,
    policy
    ):
    """
    Starts with an empty board and runs MCTS for every node traversed.
    Experiences are stored in a buffer for the neural network to be trained.
    Returns a list of experiences in a list [state, value, probability]
    """
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


class AlphaZeroTraining:
    """
    Runs the AlphaZero training by launching generations of self-play
    and training a neural network for a number of epochs after each self-play
    The network used for generating self-play game is replaced by either the latest network post-training or by the winner of the competition between the old network and the current network post-training.
    """

    def __init__(
        self,
        game_settings,
        game_training_settings,
        mcts_settings,
        nn_training_settings,
        benchmark_competition_settings,
        play_settings,
        policy,
        log_data,
    ):
        self.game_settings = game_settings
        self.game_training_settings = game_training_settings
        self.mcts_settings = mcts_settings
        self.nn_training_settings = nn_training_settings
        self.benchmark_competition_settings = benchmark_competition_settings
        self.play_settings = play_settings
        self.policy = policy
        self.log_data = log_data
        
        self.temp_policy_path = "ai_temp_ckp.pth"

    def training_pipeline(self, buffer):
        """
        Executes AlphaZero training algorithm
        1 generation includes:
         - x iterations of self-play
         - a number of epochs of neural network training
         - benchmark against a baseline
         - competition against old network
        """

        generations = self.game_training_settings.generations
        self_play_iterations = self.game_training_settings.self_play_iterations
        data_augmentation_times = self.game_training_settings.data_augmentation_times
        batch_size = self.nn_training_settings.batch_size
        temp_policy = policy_mod.Policy(
            self.temp_policy_path, 
            self.nn_training_settings, 
            self.log_data
            )
        net_compet_threshold = self.benchmark_competition_settings.net_compet_threshold
        compet_freq = self.benchmark_competition_settings.compet_freq

        for gen in trange(generations, desc = 'Generations'):
            for _ in range(self_play_iterations):
                new_exp = execute_self_play(
                    self.game_settings,
                    self.mcts_settings,
                    self.policy,
                )
                for exp in new_exp:
                    for _ in range(data_augmentation_times):
                        buffer.add(self.data_augmentation(exp))

            if buffer.buffer_len() == buffer.buffer_size_target:
                losses = temp_policy.nn_train(buffer, batch_size)
                self.log_data.save_data("nn_loss", gen, losses)
                temp_policy.save_weights()

                update_net = False
                improvement_score = self.net_compet(gen)
                if improvement_score is not None and improvement_score >= net_compet_threshold:
                    update_net = True

                if (update_net) or (compet_freq == 0):
                    self.policy.load_weights(self.temp_policy_path)
                    self.policy.save_weights()
                    print ("Network replaced at generation {}".format(gen))

            scores = self.benchmark(gen)
            if scores:
                self.log_data.save_data("compet", gen, [scores])

    def net_compet(self, gen):
        """
        Implementation 1 (not used for Tic-Tac-Toe): Measures the performance of the improved MCTS-policy
        vs. the current MCTS-policy used for self-play generations

        Implementation 2 : Measures the performance of the trained network vs. the current netowrk 
        used for self-play generations (no MCTS used). This is better for Tic-Tac-Toe even if the networks
        always play the same positions.
        """

        compet_freq = self.benchmark_competition_settings.compet_freq
        compet_rounds = self.benchmark_competition_settings.compet_rounds
        reversed = False

        if compet_freq != 0 and (gen + 1) % compet_freq == 0:
            '''
            # Competition between the 2 networks + MCTS (useless for Tic-Tac-Toe)
            new_agent = policy_player_mcts
            old_agent = policy_player_mcts
            # TO DO
            improvement_score = match_ai(
                self.game_settings,
                self.play_settings,
                new_agent,
                old_agent,
                total_rounds=compet_rounds,
            )
            '''
            scores = 0
            p1_params = {"policy_path": self.temp_policy_path, "nn_training_settings": self.nn_training_settings}
            p2_params = {"policy_path": self.nn_training_settings.policy_path, "nn_training_settings": self.nn_training_settings}

            for round_n in range(compet_rounds):
                if round_n > compet_rounds / 2 -1 :
                    reversed = True
                match_params = {"player1": [net_player, p1_params], "player2": [net_player, p2_params], "inverse_order": reversed}
                score = match_net_mcts(
                    self.game_settings,
                    self.benchmark_competition_settings,
                    **match_params
                )
                scores += score

            return scores - compet_rounds * 1

    def benchmark(self, gen):
        """
        Measures the performance of the network against a plain MCTS
        """

        benchmark_freq = self.benchmark_competition_settings.benchmark_freq
        benchmark_rounds = self.benchmark_competition_settings.benchmark_rounds
        reversed = False
        scores = 0
        p1_params = {"policy_path": self.temp_policy_path, "nn_training_settings": self.nn_training_settings}
        p2_params = {}

        if benchmark_freq != 0 and (gen + 1) % benchmark_freq == 0:
            for round_n in range(benchmark_rounds):
                if round_n > benchmark_rounds / 2 -1 :
                    reversed = True
                match_params = {"player1": [net_player, p1_params], "player2": [mcts_player, p2_params], "inverse_order": reversed}
                score = match_net_mcts(
                    self.game_settings,
                    self.benchmark_competition_settings,
                    **match_params
                )
                scores += score
            return scores / benchmark_rounds

    def data_augmentation(self, exp):

        input_board, v, prob = exp
        t, tinv = random.choice(self.transformations())
        prob = prob.reshape(3, 3)
        return t(input_board), v, tinv(prob).reshape(1, 9).squeeze(0)

    def flip(self, x, dim):

        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(
            x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
        )
        return x[tuple(indices)]

    def transformations(self):
        """
        Returns a list of transformation functions for exploiting symetries

        #################IMPROVEMENTS###################
         - review symetries 5 and 6
        """

        # transformations
        t0 = lambda x: x
        t1 = lambda x: x[:, ::-1].copy()
        t2 = lambda x: x[::-1, :].copy()
        t3 = lambda x: x[::-1, ::-1].copy()
        t4 = lambda x: x.T
        # TO DO
        t5 = lambda x: x[:, ::-1].T.copy()
        t6 = lambda x: x[::-1, :].T.copy()
        t7 = lambda x: x[::-1, ::-1].T.copy()

        tlist = [t0, t1, t2, t3, t4, t7]
        tlist_half = [t0, t1, t2, t3]

        # inverse transformations
        t0inv = lambda x: x
        t1inv = lambda x: self.flip(x, 1)
        t2inv = lambda x: self.flip(x, 0)
        t3inv = lambda x: self.flip(self.flip(x, 0), 1)
        t4inv = lambda x: x.t()
        # TO DO
        t5inv = lambda x: self.flip(x, 1).t()
        t6inv = lambda x: self.flip(x, 0).t()
        t7inv = lambda x: self.flip(self.flip(x, 0), 1).t()

        tinvlist = [t0inv, t1inv, t2inv, t3inv, t4inv, t7inv]
        transformation_list = list(zip(tlist, tinvlist))
        return transformation_list
