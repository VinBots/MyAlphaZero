from copy import copy
from math import *
import random
import time

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Node:
    """
    Implements the MCTS by building a search tree.
    A tree is made of several nodes connected between each other
    A node is defined by a game, a mother and a probability.

    A node can have children, stored as a dictionary in self.child
    self.child = {action1: Node(game1, self, p1), action2: Node(game2, self, p2), ...}
    create_child() creates self.child by iterating over possible actions from a given node
    """

    def __init__(self, game, mother=None, prob=torch.tensor(0.0, dtype=torch.float)):
        self.game = game
        self.child = {}
        self.U = 0

        self.U_sa = 0
        self.Q_sa = 0
        self.W_sa = 0

        self.prob = prob
        self.nn_v = torch.tensor(0.0, dtype=torch.float)
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

    def create_child(self, actions, probs):
        """
        create_child() creates self.child by iterating over possible actions from a given node
        """

        games = [copy(self.game) for a in actions]
        for action, game in zip(actions, games):
            game.move(action)

        child = {tuple(a): Node(g, self, p) for a, g, p in zip(actions, games, probs)}
        self.child = child

    def explore(self, policy, dir_eps, dir_alpha, dirichlet_enabled = False):
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
            if max_U == -float("inf"):
                current.U = float("inf")
                current.V = 1.0
                break
            elif max_U == float("inf"):
                current.U = -float("inf")
                current.V = -1.0
                break

            current = child[action]

        # Node has not been expanded
        if not current.child and current.outcome is None:
            # policy outputs results from the perspective of the next player
            # thus extra - sign is needed

            input = (
                torch.tensor(
                    current.game.state * current.game.player,
                    dtype=torch.float,
                    device=device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            v, probs = policy.forward_batch(input, dim_value = 0)
            current.V = -float(v.squeeze().squeeze())
            
            mask = torch.tensor(current.game.available_mask())
            probs = self.normalize(probs.view(3,3)[mask].view(-1))
            if dirichlet_enabled:                
                probs = self.dirichlet_noise(probs, dir_eps, dir_alpha)
            
            next_actions = current.game.available_moves()
            current.create_child(next_actions, probs)
            
        current.N += 1

        # Updates U and back-prop
        while current.mother:
            mother = current.mother
            mother.N += 1
            # between mother and child, the player is switched, extra - sign
            mother.V += (-current.V - mother.V) / mother.N

            # update U for all sibling nodes
            for sibling in mother.child.values():
                if sibling.U is not float("inf") and sibling.U is not -float("inf"):
                    sibling.U = sibling.V + self.c_puct * float(sibling.prob) * sqrt(
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
        child = self.child

        max_U = max(c.U for c in child.values())

        if max_U == float("inf"):
            prob = torch.tensor(
                [1.0 if c.U == float("inf") else 0 for c in child.values()]
            )
            prob_choice = prob

        else:
            # use max for numerical stability
            totalN = max(node.N for node in child.values()) + 1

            # totalN = sum(node.N for node in child.values()) + 1
            prob = torch.tensor(
                [(node.N / totalN) for node in child.values()],
                device=device,
            )

            prob_choice = torch.tensor(
                [(node.N / totalN) ** (1 / temperature) for node in child.values()],
                device=device,
            )
            
        prob = self.normalize(prob)
        prob_choice = self.normalize(prob_choice)

        nn_prob = torch.stack([node.prob for node in child.values()]).to(device)         

        nextstate = random.choices(list(child.values()), weights=prob_choice)[0]

        prob = self.game.unmask(prob)

        return nextstate, (new_state, -self.V, prob)

    def normalize(self, prob):
        """
        Normalize a tensor into a probability distribution
        """
        if torch.sum(prob) > 0:
            prob /= torch.sum(prob)

        # if sum is zero, just make things random
        else:
            num_elements = torch.numel(prob)
            prob = torch.tensor(1.0 / num_elements).repeat(num_elements)

        return prob

    def detach_mother(self):
        del self.mother
        self.mother = None

    def dirichlet_noise(self, probs, dir_eps, dir_alpha):
        
        noise = torch.distributions.Dirichlet(torch.tensor([dir_alpha] * len(probs))).sample()
        return (1-dir_eps) * probs + dir_eps * noise
    
        