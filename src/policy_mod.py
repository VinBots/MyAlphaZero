import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Policy(nn.Module):
    def __init__(self, path, nn_training_settings=None, log_data=None):
        super(Policy, self).__init__()
        self.path = path
        self.nn_training_settings = nn_training_settings
        self.log_data = log_data

        torch.set_default_dtype(torch.float32)
        self.kernel_dim = 2
        self.stride = 1
        depth = 16

        self.conv = nn.Conv2d(
            1, depth, kernel_size=self.kernel_dim, stride=self.stride, bias=False
        )
        self.size = self.kernel_dim * self.kernel_dim * depth
        self.fc1 = nn.Linear(self.size, 32)
        self.fc2 = nn.Linear(32, 32)

        # layers for the policy function
        self.fc_action1 = nn.Linear(32, 16)
        self.fc_action2 = nn.Linear(16, 9)

        # layers for the value function
        self.fc_value1 = nn.Linear(32, 8)
        self.fc_value2 = nn.Linear(8, 1)
        self.tanh_value = nn.Tanh()

        self.train_steps_count = 0

    def forward_batch(self, x, dim_value=1):
        """
        Forward pass of the tensor x
        """

        y = F.relu(self.conv(x))
        y = y.view(-1, self.size)
        y = self.fc1(y)
        y = F.relu(self.fc2(y))

        # the action head
        a = F.relu(self.fc_action1(y))
        a = self.fc_action2(a)

        mask = self.legal_actions_mask(x)
        a = mask * a

        maxa = torch.max(a)
        prob = F.softmax(a - maxa, dim=dim_value)

        value = F.relu(self.fc_value1(y))
        value = self.tanh_value(self.fc_value2(value))

        return value, prob

    def nn_train(self, replay_buffer, batch_size):
        """
        Trains the network by mini-batch gradient descent for a number of epochs
        """

        buffer_size = replay_buffer.buffer_len()
        losses = 0
        value_losses = 0
        prob_losses = 0

        self.train()
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.nn_training_settings.lr,
            weight_decay=self.nn_training_settings.weight_decay,
        )

        all_states, all_target_v, all_target_p = replay_buffer.stack(
            replay_buffer.memory.items()
        )
        training_steps = 0
        for epoch in range(self.nn_training_settings.n_epochs):

            permutation = torch.randperm(buffer_size)

            for i in range(0, buffer_size, batch_size):

                training_steps += 1
                self.train_steps_count += 1
                indices = permutation[i : i + batch_size]
                states, target_v, target_p = (
                    all_states[indices],
                    all_target_v[indices],
                    all_target_p[indices],
                )
                optimizer.zero_grad()
                pred_v, pred_p = self.forward_batch(torch.tensor(states))

                loss, value_loss, prob_loss = self.loss_function(
                    torch.tensor(states), pred_v, pred_p, target_v, target_p
                )

                loss.backward()
                self.log_data.save_grads(
                    self.train_steps_count, self.named_parameters()
                )

                optimizer.step()
                losses += float(loss)
                value_losses += float(value_loss)
                prob_losses += float(prob_loss)

                del loss

        return [
            losses / training_steps,
            value_losses / training_steps,
            prob_losses / training_steps,
        ]

    def loss_function(self, input_board, pred_v, pred_p, target_v, target_p):

        value_loss = (pred_v - target_v) ** 2

        loglist = torch.where(
            pred_p > 0, torch.log(pred_p) * target_p, torch.tensor(0.0)
        )
        policy_loss = torch.sum(loglist, dim=1, keepdim=True)

        constant_list = torch.where(
            target_p > 0, torch.log(target_p) * target_p, torch.tensor(0.0)
        )
        constant = torch.sum(constant_list, dim=1, keepdim=True)

        value_loss = value_loss.sum()
        prob_loss = -policy_loss.sum() + constant.sum()
        loss = value_loss + prob_loss

        return (
            loss / pred_p.size()[0],
            value_loss / pred_p.size()[0],
            prob_loss / pred_p.size()[0],
        )

    def save_weights(self):
        """
        Saves the network checkpoints
        """
        full_path = self.nn_training_settings.ckp_folder + "/" + self.path
        torch.save(self.state_dict(), full_path)

    def load_weights(self, path):
        """
        Loads weights of the network saved in path
        """
        full_path = self.nn_training_settings.ckp_folder + "/" + path
        self.load_state_dict(torch.load(full_path))

    def legal_actions_mask(self, x):
        """
        Returns a mask over legal actions
        """

        mask1 = (torch.abs(x) != 1).type(torch.FloatTensor).view(-1, 9)
        mask2 = torch.where(mask1 > 0, mask1, torch.tensor(1e-10))
        return mask2
