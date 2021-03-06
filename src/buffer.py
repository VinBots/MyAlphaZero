######################################################
#
# AlphaZero algorithm applied to Tic-Tac-Toe
# written by Vincent Manier (vmanier2020@gmail.com)
#
# This module implements a centralized replay buffer
# based on a dictionary (self.memory). It also includes
# a function to deduplicate similar positions.
#
######################################################


# Import libraries
import random
from collections import namedtuple
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self, buffer_size_target, batch_size, log_data, seed=0, deduplicate=False
    ):
        """
        Initializes a ReplayBuffer object
        """

        self.buffer_size_target = buffer_size_target
        self.batch_size = batch_size
        self.log_data = log_data
        self.seed = random.seed(seed)
        self.memory = {}
        self.all_fields_names = ["v", "p", "count_exp"]
        self.record_length = len(self.all_fields_names)
        self.experience = namedtuple("Experience", field_names=self.all_fields_names)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.save_keys_list = []
        self.add_counter = 0

    def add(self, exp):
        """
        Add a new experience to memory
        An experience is a namedtuple, made of a board state, v and p
        """

        state, v, p = exp
        hash_state = tuple(map(tuple, state))
        hash_exp = [hash_state, v, p]
        self.deduplicate(hash_exp)
        self.remove_keys()
        self.add_counter += 1
        self.log_data.save_data("buffer", self.add_counter, self.dist_outcomes())

    def sample(self):
        """
        Randomly samples a batch of experiences from memory
        """

        experiences = random.sample(self.memory.items(), k=self.batch_size)
        return experiences

    def stack(self, samples):
        """
        Stacks the states, p, v in tensors so that it can be ingested for neural network training
        """

        all_states = torch.stack(
            [
                torch.tensor([e[0]], dtype=torch.float, device=device)
                for e in samples
                if e is not None
            ]
        )
        all_v = torch.stack(
            [torch.tensor([e[1][0]], dtype=torch.float, device=device) for e in samples]
        )
        all_p = torch.stack([e[1][1] for e in samples])
        return (all_states, all_v, all_p)

    def buffer_len(self):
        """Return the current size of the buffer."""

        return len(self.memory)

    def dist_outcomes(self):
        """
        Returns stats about the content of the buffer
        There must be a bette way...
        """

        wins = 0
        losses = 0
        draws = 0
        for _, v in self.memory.items():
            if v[0] > 0:
                wins += 1
            elif v[0] < 0:
                losses += 1
            elif v[0] == 0:
                draws += 1
        return [wins, losses, draws]

    def deduplicate(self, exp):
        """
        Averages v and p for positions already present in the buffer
        """

        if exp[0] in self.memory:
            existing_v, existing_p, existing_count = self.memory[exp[0]]

            self.memory[exp[0]] = (
                self.average_v_p(exp[1], existing_v, existing_count + 1),
                self.average_v_p(exp[2], existing_p, existing_count + 1),
                existing_count + 1,
            )
        else:
            self.add_keys(exp[0])
            self.memory[exp[0]] = (exp[1], exp[2], 1)

    def average_v_p(self, x1, x2, count_num):
        return x2 + (x1 - x2) / count_num

    def remove_keys(self):
        """Removes items from the buffer if its size exceeds the target limit"""

        while len(self.memory) > self.buffer_size_target:
            self.memory.pop(self.save_keys_list[0])
            self.save_keys_list.pop(0)

    def add_keys(self, key_value):
        """Stores the position in the buffer in the save_keys_list"""

        self.save_keys_list.append(key_value)
