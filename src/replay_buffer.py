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

    def __init__(self, buffer_size, batch_size, seed=0):
        """
        Initializes a ReplayBuffer object
        """
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.all_fields_names = ["state", "v", "p"]
        self.record_length = len(self.all_fields_names)
        self.experience = namedtuple("Experience", \
            field_names=self.all_fields_names)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() \
            else "cpu")

    def add(self, exp):
        """
        Add a new experience to memory
        An experience is a namedtuple, made of a board state, v and p
        """
        e = self.experience._make(exp)
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

        all_states = torch.tensor(np.vstack([[[e.state] for e in samples if e is not None]]), 
                                  dtype=torch.float, device=device)
        all_v = torch.tensor(np.vstack([e.v for e in samples if e is not None]), 
                             dtype=torch.float, device=device)
        all_p = torch.tensor(np.vstack([e.p for e in samples if e is not None]), 
                             dtype=torch.float, device=device)
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
                wins+=1
            elif exp.v == -1:
                losses+=1
            elif exp.v == 0:
                draws+=1
        return (wins, losses, draws)