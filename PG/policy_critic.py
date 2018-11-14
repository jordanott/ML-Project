import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    Value Function approximator.
    """

    def __init__(self, input_size):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(input_size, 1)
        #self.l2 = nn.Linear(256, 256)
        #self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        state = torch.Tensor(state)
        #x = F.relu(self.l1(state))
        #x = F.relu(self.l2(x))
        value_estimate = self.l1(state)

        return value_estimate
