import time
import numpy as np
import random

import torch.optim as optim
import torch
import progressbar as pb

import mcts
import games_mod
from self_play import execute_self_play, fake_play
from net_utils import plot_grad_flow
from test1 import test_final_positions


class AlphaZeroTraining:
    """
    Runs the AlphaZero training by launching episods of self-play
    and training a neural network after each self-play
    """

    def __init__(
        self, game_settings, game_training_settings, nn_training_settings, policy
    ):
        self.game_settings = game_settings
        self.game_training_settings = game_training_settings
        self.nn_training_settings = nn_training_settings
        self.policy = policy
        self.vterm = []
        self.logterm = []
        self.print_every = 50  # print stats every x episods
        self.losses = []
        self.self_play_time = []
        self.training_time = []
        self.mcts_explore_time = []
        self.count_steps = []

    def training_pipeline(self, buffer):
        """
        Executes AlphaZero training algorithm
        """

        losses_list = []
        episods = self.game_training_settings.episods
        explore_steps = self.game_training_settings.explore_steps
        temp = 1.0
        self_play_iterations = self.game_training_settings.self_play_iterations
        batch_size = self.nn_training_settings.batch_size

        for ep in range(episods):
            for e in range(self_play_iterations):
                if e > self.game_training_settings.temp_threshold[0]:
                    temp = self.game_training_settings.temp_threshold[1]
                new_exp = execute_self_play(
                    self.game_settings, explore_steps, self.policy, temp
                )
                # print (new_exp)
                for i in range(len(new_exp)):
                    new_aug_exp = self.data_augmentation(new_exp[i])
                    buffer.add(new_aug_exp)

            if buffer.buffer_len() == buffer.buffer_size:
                losses, plt = self.policy.nn_train(buffer, batch_size)
                losses_list.append(losses)

            if (ep + 1) % self.print_every == 0:
                self.show_stats(ep, losses_list, plt, buffer)

        self.policy.save_weights()
        return losses_list

    def show_stats(self, ep, losses_list, plt, buffer):
        print(
            "Loss - Moving average last 10 {}".format(
                np.array(losses_list[-10:]).mean()
            )
        )
        print("Loss - last 10 {}".format(np.array(losses_list[-10:])))
        print("------------------------")
        # plt = plot_grad_flow(self.policy.named_parameters())
        # plt.savefig("nn_perf/gradients " + str(ep+1) + ".png")
        # plt.show()
        print("------------------------")

        wins, losses, draws = buffer.dist_outcomes()
        total_outcomes = wins + losses + draws
        print(
            "Wins % : {:.2%}, Losses % : {:.2%}, Draws % : {:.2%}".format(
                wins / total_outcomes, losses / total_outcomes, draws / total_outcomes
            )
        )
        test_final_positions(buffer)

    def data_augmentation(self, exp):

        # for square board, add rotations as well
        input_board, v, prob = exp
        t, tinv = random.choice(self.transformations(half=False))
        prob = prob.reshape(3, 3)
        return t(input_board), v, tinv(prob).reshape(1, 9).squeeze(0)

    def flip(self, x, dim):

        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(
            x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
        )
        return x[tuple(indices)]

    def transformations(self, half=False):
        """
        Returns a list of transformation functions for exploiting symetries
        """
        # transformations
        t0 = lambda x: x
        t1 = lambda x: x[:, ::-1].copy()
        t2 = lambda x: x[::-1, :].copy()
        t3 = lambda x: x[::-1, ::-1].copy()
        t4 = lambda x: x.T
        # t5 = lambda x: x[:, ::-1].T.copy()
        # t6 = lambda x: x[::-1, :].T.copy()
        t7 = lambda x: x[::-1, ::-1].T.copy()

        tlist = [t0, t1, t2, t3, t4, t7]
        tlist_half = [t0, t1, t2, t3]

        # inverse transformations
        t0inv = lambda x: x
        t1inv = lambda x: self.flip(x, 1)
        t2inv = lambda x: self.flip(x, 0)
        t3inv = lambda x: self.flip(self.flip(x, 0), 1)
        t4inv = lambda x: x.t()
        # t5inv = lambda x: self.flip(x, 0).t()
        # t6inv = lambda x: self.flip(x, 1).t()
        t7inv = lambda x: self.flip(self.flip(x, 0), 1).t()

        if half:
            tinvlist_half = [t0inv, t1inv, t2inv, t3inv]
            transformation_list_half = list(zip(tlist_half, tinvlist_half))
            return transformation_list_half

        else:
            tinvlist = [t0inv, t1inv, t2inv, t3inv, t4inv, t7inv]
            transformation_list = list(zip(tlist, tinvlist))
            return transformation_list
