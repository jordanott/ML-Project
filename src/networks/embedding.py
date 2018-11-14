import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size=1):
        super(CNN, self).__init__()
        ks = [3, 3, 3, 3, 3, 3, 2]; ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]; nm = [64, 128, 256, 256, 512]

        self.cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = 1 if i == 0 else nm[i - 1]
            nOut = nm[i]
            self.cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                self.cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            self.cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))

        convRelu(0)
        self.cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        convRelu(1)
        self.cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        convRelu(2, True)
        convRelu(3)
        self.cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2)))
        convRelu(4, True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.cnn(x)
        return x
