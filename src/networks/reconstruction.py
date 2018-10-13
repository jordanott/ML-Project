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


class FF(nn.Module):
    def __init__(self,input_size,output_size):
        super(FF, self).__init__()
        self.l1 = nn.Linear(input_size,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,128)
        self.l4 = nn.Linear(128,output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)

        return x
