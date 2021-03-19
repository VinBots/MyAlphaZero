import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from net_utils import plot_grad_flow


class Policy(nn.Module):
    def __init__(self, nn_training_settings=None):
        super(Policy, self).__init__()
        depth = 16
        self.nn_training_settings = nn_training_settings

        self.conv = nn.Conv2d(1, depth, kernel_size=2, stride=1, bias=False)
        self.size = 2 * 2 * depth
        self.fc1 = nn.Linear(self.size, 32)
        self.fc2 = nn.Linear(32, 32)

        # layers for the policy function
        self.fc_action1 = nn.Linear(32, 16)
        self.fc_action2 = nn.Linear(16, 9)

        # layers for the value function
        self.fc_value1 = nn.Linear(32, 8)
        self.fc_value2 = nn.Linear(8, 1)
        self.tanh_value = nn.Tanh()
        self.store_counter = 0
        self.all_parameters = {}
        self.loss_records = []

    def forward(self, x):

        y = F.relu(self.conv(x))
        y = y.view(-1, self.size)
        y = self.fc1(y)
        y = F.relu(self.fc2(y))
        # the action head
        a = F.relu(self.fc_action1(y))
        a = self.fc_action2(a)

        # availability of moves
        avail = (torch.abs(x.squeeze()) != 1).type(torch.FloatTensor)
        avail = avail.view(-1, 9)

        # locations where actions are not possible, we set the prob to zero
        maxa = torch.max(a)
        # subtract off max for numerical stability (avoids blowing up at infinity)
        exp = avail * torch.exp(a - maxa)
        prob = exp / torch.sum(exp)

        # the value head
        value = F.relu(self.fc_value1(y))
        value = self.tanh_value(self.fc_value2(value))

        return prob.view(3, 3), value
        # return prob, value

    def forward_batch(self, x):

        y = F.relu(self.conv(x))
        y = y.view(-1, self.size)

        y = self.fc1(y)
        y = F.relu(self.fc2(y))

        # the action head
        a = F.relu(self.fc_action1(y))
        a = self.fc_action2(a)

        maxa = torch.max(a)
        mask = self.legal_actions_mask(x)
        exp = mask * torch.exp(a - maxa)
        prob = exp / torch.sum(exp, 1, keepdim=True)
        # print ("Probabilities vectors: {}".format(prob))
        # print ("board positions: {}".format(x))

        # the value head
        value = F.relu(self.fc_value1(y))
        value = self.tanh_value(self.fc_value2(value))

        return value, prob

    def nn_train(self, replay_buffer, batch_size):
        """
        Calculates the loss function and applies gradient descent to minimize it
        """
        training_steps = self.nn_training_settings.training_steps
        losses = 0

        self.train()
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.nn_training_settings.lr,
            weight_decay=self.nn_training_settings.weight_decay,
        )

        for i in range(training_steps):
            all_states, target_v, target_p = replay_buffer.stack(replay_buffer.sample())
            optimizer.zero_grad()
            pred_v, pred_p = self.forward_batch(torch.Tensor(all_states))

            loss = self.loss_function(
                torch.Tensor(all_states), pred_v, pred_p, target_v, target_p
            )
            loss.backward()
            # plt = plot_grad_flow(self.named_parameters())
            optimizer.step()
            losses += float(loss)
            del loss

        self.store_parameters()

        # Print model's state_dict
        # print("Model's state_dict:")
        # for param_tensor in self.state_dict():
        #    print(param_tensor, "\t", self.state_dict()[param_tensor])
        # for name, param in self.named_parameters():
        #    print(name, param.grad.norm())

        # Print optimizer's state_dict
        # print("Optimizer's state_dict:")
        # for var_name in optimizer.state_dict():
        # print(var_name, "\t", optimizer.state_dict()[var_name])
        return float(losses / training_steps), plt

    def loss_function(self, input_board, pred_v, pred_p, target_v, target_p):

        # print (target_v)
        # vlue_loss = ((pred_v - target_v)**2).sum()
        value_loss = (pred_v - target_v) ** 2
        # print ("value_loss: {}".format(value_loss))

        loglist = torch.where(
            pred_p > 0, torch.log(pred_p) * target_p, torch.tensor(0.0)
        )
        policy_loss = torch.sum(loglist, dim=1, keepdim=True)
        # print ("policy_loss = {}".format(policy_loss))

        constant_list = torch.where(
            target_p > 0, torch.log(target_p) * target_p, torch.tensor(0.0)
        )
        constant = torch.sum(constant_list, dim=1, keepdim=True)
        # print ("constant: {}".format(constant))

        loss = value_loss - policy_loss + constant
        loss = loss.squeeze().sum()
        # print ("loss = {}".format(loss))

        self.loss_records.append(
            np.array([value_loss.sum(), -policy_loss.sum(), constant.sum()])
            / pred_p.size()[0]
        )

        return loss / pred_p.size()[0]

    def save_weights(self):
        torch.save(self.state_dict(), "ai_ckp.pth")

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        # self.eval()
        # print ("Models: {} loaded...".format(path))

    def store_parameters(self):
        """
        Stores the parameters of the model
        """
        self.store_counter += 1
        self.all_parameters[self.store_counter] = self.parameters()

    def legal_actions_mask(self, x):
        """
        Returns a mask over legal actions
        """
        mask1 = (torch.abs(x) != 1).type(torch.FloatTensor).view(-1, 9)
        mask2 = torch.where(mask1 > 0, mask1, torch.tensor(1e-10))
        return mask2
