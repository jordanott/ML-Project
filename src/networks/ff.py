import torch
import torch.nn as nn
import torch.nn.functional as F

class FF(nn.Module):
    def __init__(self, input_size, num_actions):
        super(FF, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

'''
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
'''
