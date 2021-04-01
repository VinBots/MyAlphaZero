from copy import copy
from math import *
import random
import time

import numpy as np


class Node:
    """
    Implements the MCTS by building a search tree.
    A tree is made of several nodes connected between each other
    A node is defined by a game, a mother and a probability.

    A node can have children, stored as a dictionary in self.child
    self.child = {action1: Node(game1, self, p1), action2: Node(game2, self, p2), ...}
    create_child() creates self.child by iterating over possible actions from a given node
    """

    def __init__(self, game, mother=None, prob = None):
        self.game = game
        self.child = {}
        self.U = 0

        self.U_sa = 0
        self.Q_sa = 0
        self.W_sa = 0

        self.prob = prob
        self.N = 0
        self.V = 0
        self.outcome = self.game.score
        self.c_puct = 1.0

        # if game is won/loss/draw
        if self.game.score is not None:
            self.V = self.game.score * self.game.player
            self.U = 0 if self.game.score is 0 else self.V * float("inf")
        # link to previous node
        self.mother = mother

    def create_child(self, actions):
        """
        create_child() creates self.child by iterating over possible actions from a given node
        """

        games = [copy(self.game) for a in actions]
        for action, game in zip(actions, games):
            game.move(action)

        child = {tuple(a): Node(g, self) for a, g in zip(actions, games)}
        self.child = child

    def explore(self):
        """
        Implements the expansion, simulation and backpropagation steps
        This method should be further split
        """

        if self.game.score is not None:
            raise ValueError("game has ended with score {0:d}".format(self.game.score))
        current = self

        while current.child and current.outcome is None:

            child = current.child
            max_U = max(c.U for c in child.values())
            actions = [a for a, c in child.items() if c.U == max_U]

            if len(actions) == 0:
                print("error zero length ", max_U)
                print(current.game.state)
                actions = [a for a, c in child.items()]

            action = random.choice(actions)
            current = child[action]

        # Node has not been expanded
        if not current.child and current.outcome is None:

            current.V = -current.game.player * self.roll_out(current.game)
            #print ("Current player {}; V {}".format(
                #current.game.player, current.V))
            next_actions = current.game.available_moves()
            current.create_child(next_actions)
            
        current.N += 1

        # Updates U and back-prop
        while current.mother:
            mother = current.mother
            mother.N += 1
            # beteen mother and child, the player is switched, extra - sign
            mother.V += (-current.V - mother.V) / mother.N

            # update U for all sibling nodes
            for sibling in mother.child.values():
                if sibling.U is not float("inf") and sibling.U is not -float("inf"):
                    sibling.U = sibling.V + self.c_puct * sqrt(
                        mother.N
                    ) / (1 + sibling.N)
            current = current.mother

    def next(self, temperature=1.0):
        """
        Plays an action according to stats collected in the explore stage
        """
        new_state = self.game.state

        if self.game.score is not None:
            raise ValueError("game has ended with score {0:d}".format(self.game.score))
        if not self.child:
            print(self.game.state)
            raise ValueError("no children found and game hasn't ended")
        
        totalN = max(node.N for node in self.child.values()) + 1
        prob_choice = np.array([(node.N / totalN) ** (1 / temperature) for node in self.child.values()])
        #print ("N[4] = {}".format(list(self.child.values())[4].N))


        prob_choice = self.normalize(prob_choice)
        #print ("Probabilities = {}".format(prob_choice))
        nextstate = random.choices(list(self.child.values()), weights=prob_choice)[0]
        return nextstate

    def normalize(self, prob):
        # normalize the probability
        if np.sum(prob) > 0:
            prob /= np.sum(prob)

        # if sum is zero, just make things random
        else:
            num_elements = np.numel(prob)
            prob = np.array(1.0 / num_elements).repeat(num_elements)

        return prob

    def detach_mother(self):
        del self.mother
        self.mother = None
        
    def roll_out (self, game, nb_roll_out = 1):
        #print ("Roll out asked for game state {}".format(game.state))
        scores = []
        for _ in range(nb_roll_out):
            sim_game = copy(game)
            while sim_game.score == None:          
                random_move = random.choice(sim_game.available_moves())
                #print (random_move)
                sim_game.move(random_move)
                #print (sim_game.state)
            scores.append(sim_game.score)
        return np.average(np.array(scores))
