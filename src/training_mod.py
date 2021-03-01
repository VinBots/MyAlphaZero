from copy import copy
import random
import numpy as np

import torch.optim as optim
import torch
import progressbar as pb

import policy_mod
import mcts
import games_mod


class AlphaZeroTraining:
    """TO DO"""
    
    def __init__(self, game_settings, game_training_settings, nn_training_settings, policy):
        self.game_settings = game_settings
        self.game_training_settings = game_training_settings
        self.nn_training_settings = nn_training_settings
        self.policy = policy
        self.vterm = []
        self.logterm = []
        self.print_every = 50 #print stats every x episods
        self.losses = []
        self.training_pipeline()
        
    def training_pipeline(self):
        """TO DO"""

        outcomes = []
        losses = []
        
        max_episods = self.game_training_settings.episods
        
        widget = ['training loop: ', pb.Percentage(), ' ', 
                  pb.Bar(), ' ', pb.ETA() ]
        timer = pb.ProgressBar(widgets=widget, maxval=max_episods).start()

        for e in range(max_episods):
            self.vterm = []
            self.logterm = []            
            
            #self-play returns the outcome
            outcome = self.self_play() 
            
            # the network is trained with the experience accumulated during self-play
            policy, loss = self.policy_training(outcome) 
            
            outcomes.append(outcome)
            losses.append(float(loss))
            if (e+1) % self.print_every==0:
                self.print_stats(e, losses, outcomes)

            timer.update(e+1)

        timer.finish()
        self.policy.save_weights()
        self.losses = losses
        
    def self_play(self):
        """TO DO"""
        
        mytree = mcts.Node(games_mod.ConnectN(self.game_settings))
        
        while mytree.outcome is None:
            for _ in range(self.game_training_settings.explore_steps):
                mytree.explore(self.policy)
            
            current_player = mytree.game.player
            mytree, (v, nn_v, p, nn_p) = mytree.next()
            self.collect_experiences (current_player,v, nn_v, p, nn_p) 
            mytree.detach_mother()
            outcome = mytree.outcome
            
        return outcome
    
    def collect_experiences(self, current_player,v, nn_v, p, nn_p):
        """TO DO"""
        
        # compute prob* log pi
        loglist = torch.log(nn_p) * p
        # constant term to make sure if policy result = MCTS result, loss = 0
        constant = torch.where(p > 0, p * torch.log(p), torch.tensor(0.0))

        self.logterm.append(-torch.sum(loglist - constant))
        self.vterm.append(nn_v * current_player)    
    
    
    def policy_training(self, outcome):
        """TO DO"""

        optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.nn_training_settings.lr,
            weight_decay=self.nn_training_settings.weight_decay,
        )
        
        loss = torch.sum((torch.stack(self.vterm) - outcome) ** 2 + torch.stack(self.logterm))
        loss_value = float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss
        return (self.policy, loss_value)
    
    def print_stats(self, e, losses, outcomes):
        """TO DO"""
        
        print("game: ",e+1, ", mean loss: {:3.2f}".format(np.mean(losses[-20:])),
              ", Last 10 Results: ", outcomes[-10:])
        
    def get_losses(self):
        """TO DO"""
        
        return self.losses