"""
This module implements a centralized replay buffer. The main methods are add and sample.
"""

# Import libraries
import random
from collections import namedtuple, deque
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=0, deduplicate = False):
        """
        Initializes a ReplayBuffer object
        """

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.deduplicate = deduplicate
        self.memory = deque(maxlen=buffer_size)
        self.all_fields_names = ["state", "v", "p"]
        self.record_length = len(self.all_fields_names)
        self.experience = namedtuple("Experience", field_names=self.all_fields_names)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add(self, exp):
        """
        Add a new experience to memory
        An experience is a namedtuple, made of a board state, v and p
        """
        """
        exp = self.experience(state=np.array([[-1.,  1., -1.],
                                           [ 0.,  1.,  0.],
                                           [ 0.,  0.,  0.]]), 
                              v=1.0, 
                              p=torch.tensor([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]))
        
        """
        e = self.experience._make(exp)
        if self.deduplicate:
            self.deduplicate(e)
        else:
            self.memory.append(e)

    def sample(self):
        """
        Randomly samples a batch of experiences from memory
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        return experiences

    def stack(self, samples):
        """
        Stacks the states, p, v in tensors so that it can be ingested for neural network training
        """

        all_states = torch.tensor(
            np.vstack([[[e.state] for e in samples if e is not None]]),
            dtype=torch.float,
            device=device,
        )
        all_v = torch.tensor(
            np.vstack([e.v for e in samples if e is not None]),
            dtype=torch.float,
            device=device,
        )
        all_p = torch.tensor(
            np.vstack([e.p for e in samples if e is not None]),
            dtype=torch.float,
            device=device,
        )
        return (all_states, all_v, all_p)

    def buffer_len(self):
        """Return the current size of the buffer."""
        return len(self.memory)

    def dist_outcomes(self):
        wins = 0
        losses = 0
        draws = 0

        for exp in self.memory:
            if exp.v == 1:
                wins += 1
            elif exp.v == -1:
                losses += 1
            elif exp.v == 0:
                draws += 1
        return (wins, losses, draws)

    def deduplicate(self, exp):
        idx, existing_state = self.find_exp(exp)
        if idx is not None:
            existing_exp = self.memory[idx]
            exp = [
                exp.state,
                self.average_v(exp.v, existing_exp.v),
                self.average_p(exp.p, existing_exp.p),
            ]

            e = self.experience._make(exp)
            self.memory[idx] = e
        else:
            self.memory.append(exp)

    def find_exp(self, new_exp):
        # search = (index in buffer, namedtuple experience found)
        idx = 0
        found = False
        for state, _, _ in self.memory:
            new_exp_state = new_exp.state.astype(int)
            existing_state = state.astype(int)

            if np.array_equal(new_exp_state, existing_state):
                found = True
                break
            idx += 1
        if found:
            return (idx, existing_state)
        return (None, None)

    def average_v(self, x1, x2):
        return x2 + 0.1 * (x1-x2)

    def average_p(self, a, b):
        return b + 0.1 * (a-b)
        # return [(f + g) /2 for f,g in zip(a,b)]
