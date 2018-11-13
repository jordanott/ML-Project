import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Policy Function approximator.
    """
    def __init__(self, input_size, num_actions=4):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, num_actions)

    def forward(self, state):
        state = torch.Tensor(state)
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        dist_params = self.l3(x)
        # TODO: fix this
        self.dist = torch.distributions.normal.Normal(mu, sigma)
        action = self.dist.sample()
        return action
