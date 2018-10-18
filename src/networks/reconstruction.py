import torch
import torch.nn as nn
import torch.nn.functional as F

class dCNN(nn.Module):
    def __init__(self):
        super(dCNN, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)
        self.dconv2 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=1)
        #self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(self.dconv1(x))
        x = self.dconv2(x)

        return x
