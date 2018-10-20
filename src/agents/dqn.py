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

    def forward(self, x):
        e = self.embed(x).unsqueeze(0)
        lstm_out = self.lstm(e)

        actions = self.actions(lstm_out)

        word_prediction = self.word(lstm_out)
        word_prediction = F.log_softmax(word_prediction,dim=1)

        return actions, word_prediction

class DQN(object):
    def __init__(self,num_actions,num_words=100,VISUALIZE=False):
        # replay memory
        self.memory = []
        # discount rate
        self.gamma = 0.8
        # exploration rate
        self.epsilon = 1
        # batch size
        self.batch_size = 32
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
        # store experience
        self.memory.append(episode)

    def act(self,state):
        q_values,w = self.model(state)
        w = torch.argmax(w,dim=1)
        # sample random action with probability epsilon
        if uniform(0, 1) < self.epsilon:
            return randint(0,self.num_actions-1),w

        a = torch.argmax(q_values,dim=1).data.numpy()[0]

        return a,w

    def get_batch(self):
        # if replay memory is big enough take random sample
        if len(self.memory) > self.batch_size:
            return sample(self.memory,self.batch_size)

        return self.memory

    def replay(self, behaviour):
        batch = self.get_batch()

        for episode in batch:
            self.reset(); behaviour.lstm.reset_hidden()
            for time_step in episode:
                # get experience from batch
                s,a,r,s_prime,done,correct_word = time_step
                # Q(s, a; theta_target)
                q_s,w = self.model(s)
                q_s_a = q_s[0,a]

                target = r
                # if the round hasn't ended; estimate value of taking best action in s_prime
                if not done:
                    # max(Q(s', a'; theta_behaviour))
                    q_s_prime, w_prime = behaviour(s_prime)
                    q_s_a_prime = torch.max(q_s_prime)
                    # add to reward with discount factor gamma
                    target = r + self.gamma * q_s_a_prime

                error = target - q_s_a
                clipped_error = -1.0 * error.clamp(-1,1)

                q_s_a.backward(clipped_error, retain_graph=True)
                self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
