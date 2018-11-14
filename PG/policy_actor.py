import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Policy Function approximator.
    """
    def __init__(self, input_size, num_actions=2):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(input_size, num_actions)
        #self.l2 = nn.Linear(256, 256)
        #self.l3 = nn.Linear(256, num_actions)
        self.history = {'mu':[], 'sigma':[]}
    def forward(self, state):
        state = torch.Tensor(state)
        #x = F.relu(self.l1(state))
        #x = F.relu(self.l2(x))
        mu, sigma = self.l1(state)
        # TODO: fix this
        self.dist = torch.distributions.normal.Normal(mu, sigma)
        action = self.dist.sample()

        self.history['mu'].append(mu)
        self.history['sigma'].append(sigma)

        return action

    
