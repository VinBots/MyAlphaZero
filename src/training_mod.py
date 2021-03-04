import time
import numpy as np

import torch.optim as optim
import torch
import progressbar as pb

import mcts
import games_mod


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
        self.print_every = 500  # print stats every x episods
        self.losses = []
        self.self_play_time = []
        self.training_time = []
        self.mcts_explore_time = []
        self.count_steps = []
        self.training_pipeline()

    def training_pipeline(self):
        """TO DO"""

        outcomes = []
        losses = []
        
        self_play_time = []
        training_time = []

        max_episods = self.game_training_settings.episods
        temp = 1.0

        #widget = ["training loop: ", pb.Percentage(), " ", pb.Bar(), " ", pb.ETA()]
        #timer = pb.ProgressBar(widgets=widget, maxval=max_episods).start()

        for e in range(max_episods):
            
            self.vterm = []
            self.logterm = []

            # self-play returns the outcome
            if e > self.game_training_settings.temp_threshold[0]:
                temp = self.game_training_settings.temp_threshold[1]
            t0 = time.time()
            outcome = self.self_play(temp)
            t1 = time.time()
            self_play_time.append(t1 - t0)

            # the network is trained with the experience accumulated during self-play
            t2 = time.time()
            self.policy, loss = self.policy_training(outcome)
            t3 = time.time()
            training_time.append(t3 - t2)

            outcomes.append(outcome)
            losses.append(float(loss))
            if (e + 1) % self.print_every == 0:
                self.print_stats(e, losses, outcomes)

            #timer.update(e + 1)

        #timer.finish()
        self.policy.save_weights()
        
        self.losses = losses
        self.self_play_time = self_play_time
        self.training_time = training_time

    def self_play(self, temp):
        """
        Starts with an empty board and runs MCTS for every node traversed.
        Experiences are stored for the neural network to be trained.
        """
        count_steps = 0

        mytree = mcts.Node(games_mod.ConnectN(self.game_settings))
        mcts_explore_time = []

        while mytree.outcome is None:
            t4 = time.time()
            for _ in range(self.game_training_settings.explore_steps):
                mytree.explore(self.policy)
            t5 = time.time()
            mcts_explore_time.append(t5 - t4)
            current_player = mytree.game.player
            mytree, (v, nn_v, p, nn_p) = mytree.next(temperature = temp)
            self.store_loss_term(current_player, v, nn_v, p, nn_p)
            mytree.detach_mother()
            outcome = mytree.outcome
            count_steps+=1

        mcts_explore_time_per_self_play = np.array(mcts_explore_time).mean()
        self.mcts_explore_time.append(mcts_explore_time_per_self_play)
        self.count_steps.append(count_steps)
        return outcome

    def store_loss_term(self, current_player, v, nn_v, p, nn_p):
        """
        Calculates the v term and log term of the loss function of 
        the neural network
        """

        # compute prob * log pi
        loglist = torch.log(nn_p) * p
        # constant term to make sure if policy result = MCTS result, loss = 0
        constant = torch.where(p > 0, p * torch.log(p), torch.tensor(0.0))

        self.logterm.append(-torch.sum(loglist - constant))
        self.vterm.append(nn_v * current_player)

    def policy_training(self, outcome):
        """
        Calculates the loss function and applies gradient descent to minimize it
        """

        optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.nn_training_settings.lr,
            weight_decay=self.nn_training_settings.weight_decay,
        )

        loss = torch.sum(
            (torch.stack(self.vterm) - outcome) ** 2 + torch.stack(self.logterm)
        )
        loss_value = float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss
        return (self.policy, loss_value)

    def print_stats(self, e, losses, outcomes):
        """
        print statistics during the training process
        """

        print(
            "game: ",
            e + 1,
            ", mean loss: {:3.2f}".format(np.mean(losses[-20:])),
            ", Last 10 Results: ",
            outcomes[-10:],
        )

    def get_losses(self):
        """
        Returns a list of losses by episod of self-play training
        """
        return self.losses
    
    def get_time_stats(self):
        """
        Returns the statistics re. time measurements
        """
        
        return (self.self_play_time, self.training_time, self.mcts_explore_time, self.count_steps)
