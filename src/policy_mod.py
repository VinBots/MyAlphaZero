import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.conv = nn.Conv2d(1, 16, kernel_size=2, stride=1, bias=False)
        self.size = 2 * 2 * 16
        self.fc = nn.Linear(self.size, 32)

        # layers for the policy
        self.fc_action1 = nn.Linear(32, 16)
        self.fc_action2 = nn.Linear(16, 9)

        # layers for the critic
        self.fc_value1 = nn.Linear(32, 8)
        self.fc_value2 = nn.Linear(8, 1)
        self.tanh_value = nn.Tanh()

    def forward(self, x):

        y = F.relu(self.conv(x))
        y = y.view(-1, self.size)
        y = F.relu(self.fc(y))

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

    def save_weights(self):
        torch.save(self.state_dict(), "ai_ckp.pth")

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        # print ("Models: {} loaded...".format(path))
