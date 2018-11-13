import torch
import torch.nn as nn

class Critic(nn.Module):
    """
    Value Function approximator.
    """

    def __init__(self, input_size):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(input_size, 1)

    def forward(self, state):
        state = torch.Tensor(state)
        value_estimate = self.l1(state)

        return value_estimate
