import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.nn.functional as F

from copy import deepcopy
from src.networks.ff import FF
from src.networks.lstm import LSTM
from src.networks.embedding import CNN
from random import randint,uniform,sample

class Model(nn.Module):
    def __init__(self, num_actions, num_words):
        super(Model, self).__init__()

        self.embed = CNN()
        self.lstm = LSTM(64*29*29, hidden_size=256)
        self.word = FF(input_size=256,num_outputs=num_words)
        self.actions = FF(input_size=256,num_outputs=num_actions)
        self.frozen = False

    def freeze(self):
        self.embed.weight.requires_grad=False
        self.lstm.weight.requires_grad=False
        self.frozen = True

    def forward(self, x):
        e = self.embed(x).unsqueeze(0)
        lstm_out = self.lstm(e)

        actions = self.actions(lstm_out)
        if not self.frozen: actions = F.log_softmax(actions, dim=1)

        word_prediction = self.word(lstm_out)
        word_prediction = F.log_softmax(word_prediction, dim=1)

        return actions, word_prediction

class DQN(object):
    def __init__(self,num_actions,num_words=13551,VISUALIZE=False):
        # replay memory
        self.memory = []
        # discount rate
        self.gamma = 0.8
        # exploration rate
        self.epsilon = 1
        # batch size
        self.batch_size = 1
        # memory size
        self.memory_size = 2000
        # visualize
        self.visualize = VISUALIZE
        # minimum exploration
        self.epsilon_min = 0.05
        # decay
        self.epsilon_decay = 0.99999
        # number of environment actions
        self.num_actions = num_actions
        # load reader network
        self.model = Model(num_actions, num_words + 2)
        # model optimizer
        self.opt = opt.RMSprop(self.model.parameters())

    def reset(self):
        self.model.lstm.reset_hidden()

    def copy(self,model):
        # avoid leaf copy problem
        model.lstm.reset_hidden()
        self.model = deepcopy(model)

    def remember(self,episode):
        if len(self.memory) == self.memory_size:
            del self.memory[0]
        if episode[-1][-1][-1] is None:
            del episode[-1][-1]
            print('Deleting word focus seq...')
        # store experience
        self.memory.append(episode)

    def get_batch(self):
        # if replay memory is big enough take random sample
        if len(self.memory) > self.batch_size:
            return sample(self.memory,self.batch_size)

        return self.memory

    def replay(self):
        batch = self.get_batch()
        print(len(batch[0]))
        for episode in batch:
            self.reset()
            for word_time in episode:
                for saccade in word_time:
                    # get experience from batch
                    s,a,r,s_prime,done,correct_word = saccade
                    # Q(s, a; theta_target)
                    action_pred,word_pred = self.model(s)

                    loss = F.nll_loss(action_pred, torch.Tensor([a]).long())

                    if correct_word: # backprop word prediction
                        print(correct_word, word_pred.shape)
                        loss += F.nll_loss(word_pred, torch.Tensor([correct_word]).long())

                    loss.backward(retain_graph=True)
                    self.opt.step()
