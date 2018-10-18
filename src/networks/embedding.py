import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        #self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        return x
