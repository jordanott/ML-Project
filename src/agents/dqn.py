import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

from agent import Agent
from copy import deepcopy
from src.networks.ff import FF
from src.networks.lstm import LSTM
from src.networks.embedding import CNN
from random import randint,uniform,sample

class CharNet(nn.Module):
    def __init__(self, num_chars):
        super(CharNet, self).__init__()

        # modules for character prediction
        self.c_cnn = CNN()
        self.c_lstm = LSTM(64*512, hidden_size=256)
        self.c_ff = FF(input_size=256,num_outputs=num_chars)
        #self.IMITATE = True

    def reset_lstm(self):
        self.c_lstm.reset_hidden()

    def forward(self, x):
        c = self.c_cnn(x).unsqueeze(0)
        c = self.c_lstm(c)
        c = self.c_ff(c)

        #if not self.IMITATE: c = F.log_softmax(c, dim=1)
        return c

class ActNet(nn.Module):
    def __init__(self, num_actions):
        super(ActNet, self).__init__()

        self.a_cnn = CNN()
        self.a_lstm = LSTM(64*512, hidden_size=256)
        self.a_ff = FF(input_size=256,num_outputs=num_actions)

        self.IMITATE = True
    def forward(self, x):
        a = self.a_cnn(x).unsqueeze(0)
        a = self.a_lstm(a)
        a = self.a_ff(a)
        if self.IMITATE: a = F.log_softmax(a, dim=1)
        return a

    def reset_lstm(self):
        self.a_lstm.reset_hidden()


class DQN(Agent):
    def __init__(self,num_actions,PER_LINE=True,DIR='',num_chars=80,VISUALIZE=False):
        super(DQN, self).__init__(PER_LINE=PER_LINE, DIR=DIR)
        # replay memory
        self.memory = []; self.imitate_memory = []
        # discount rate
        self.gamma = 0.9
        # exploration rate
        self.epsilon = 0.5
        # batch size
        self.batch_size = 10
        # memory size
        self.memory_size = 200
        # visualize
        self.visualize = VISUALIZE
        # minimum exploration
        self.epsilon_min = 0.05
        # decay
        self.epsilon_decay = 0.999
        # number of environment actions
        self.num_actions = num_actions

        self.num_chars = num_chars
        # load reader network
        self.char_net = CharNet(num_chars)
        self.act_net = ActNet(num_actions)
        # model optimizer
        self.char_net_opt = opt.Adam(self.char_net.parameters(), lr=0.0002)
        self.act_net_opt = opt.Adam(self.act_net.parameters(), lr=0.0002)

        # move model to GPU
        self.char_net.cuda()
        self.act_net.cuda()

    def act(self,state):
        # sample random action with probability epsilon
        if uniform(0, 1) < self.epsilon:
            return randint(0,self.num_actions-1), randint(0, self.num_chars-1)

        state = state.cuda()
        q_values = self.act_net(state)
        char = self.char_net(state)

        w = torch.argmax(char,dim=1)
        a = torch.argmax(q_values,dim=1).cpu().data.numpy()[0]

        return a,w

    def remember(self,episode):
        if len(self.memory) == self.memory_size:
            del self.memory[0]

        self.memory.append(episode)

    def get_batch(self):
        # if replay memory is big enough take random sample
        if len(self.memory) > self.batch_size:
            return sample(self.memory,self.batch_size)

        return self.memory

    def replay(self, target):
        batch = self.get_batch()
        action_q_values = []
        for episode in batch:
            self.reset(); target.reset()
            for time_step in episode:
                # get experience from batch
                s,a,r,s_prime,done = time_step

                # move data to GPU
                s, a = s.cuda(), torch.Tensor([a]).long().cuda()

                # Q(s, a; theta_actor)
                q_s = self.act_net(s)
                q_s_a = q_s[0,a]

                # save q values
                action_q_values.append(q_s.cpu().data.numpy()[0])

                # if the episode hasn't ended; estimate value of taking best action in s_prime
                if not done:
                    s_prime =  s_prime.cuda()
                    # max(Q(s', a'; theta_target))
                    q_s_prime = target.act_net(s_prime)

                    q_s_a_prime = torch.max(q_s_prime)
                    # add to reward with discount factor gamma
                    r += self.gamma * q_s_a_prime

                error = r - q_s_a
                clipped_error = -1.0 * error.clamp(-1,1)

                q_s_a.backward(clipped_error, retain_graph=True)
                self.act_net_opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action_q_values
