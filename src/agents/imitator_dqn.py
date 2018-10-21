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
        # GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # move model to GPU
        self.model = self.model.to(self.device)

    def reset(self):
        self.model.lstm.reset_hidden()

    def copy(self,model):
        # avoid leaf copy problem
        model.lstm.reset_hidden()
        self.model = deepcopy(model)

    def remember(self,episode):
        deleted = 0
        if len(self.memory) == self.memory_size:
            del self.memory[0]
        for idx in range(len(episode)):
            if episode[idx][-1][-1] is None:
                del episode[idx]; deleted += 1

        print('Word focuses deleted:', deleted)
        # store experience
        self.memory.append(episode)

    def get_batch(self):
        # if replay memory is big enough take random sample
        if len(self.memory) > self.batch_size:
            return sample(self.memory,self.batch_size)

        return self.memory

    def replay(self):
        total_loss = {'action_loss': [], 'word_loss':[]}
        batch = self.get_batch()
        for episode in batch: # should only be one episode per batch
            for word_focus in episode:
                self.reset() # reset hidden states
                action_loss_history = []

                for saccade in word_focus:
                    action_loss, word_loss = 0,0
                    # get experience from saccade
                    s,a,r,s_prime,done,correct_word = saccade

                    # move data to GPU
                    s, a = s.to(self.device), torch.Tensor([a]).long().to(self.device)

                    # network predicted action and word
                    action_pred,word_pred = self.model(s)

                    # action loss
                    action_loss = F.nll_loss(action_pred, a)
                    # record action loss
                    action_loss_history.append(action_loss.item())

                    if correct_word: # if its time to predict a word
                        word_pred = word_pred.to(self.device)
                        correct_word = torch.Tensor([correct_word]).long().to(self.device)

                        word_loss = F.nll_loss(word_pred, correct_word)
                        # record losses
                        total_loss['action_loss'].append(np.mean(action_loss_history))
                        total_loss['word_loss'].append(word_loss.item())
                        action_loss_history = []

                    # total loss: combination of action and word losses
                    loss = action_loss + word_loss

                    loss.backward(retain_graph=True)
                    self.opt.step()
        return total_loss
