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

class Model(nn.Module):
    def __init__(self, num_actions, num_chars):
        super(Model, self).__init__()
        # modules for action prediction
        self.a_cnn = CNN()
        self.a_lstm = LSTM(64*29*29, hidden_size=256)
        self.a_ff = FF(input_size=256,num_outputs=num_actions)

        # modules for character prediction
        self.c_cnn = CNN()
        self.c_lstm = LSTM(64*29*29, hidden_size=256)
        self.c_ff = FF(input_size=256,num_outputs=num_chars)

        self.IMITATE = True

    def reset_lstm(self):
        self.c_lstm.reset_hidden()
        self.a_lstm.reset_hidden()

    def forward(self, x):
        a = self.a_cnn(x).unsqueeze(0)
        a = self.a_lstm(a)
        a = self.a_ff(a)

        c = self.c_cnn(x).unsqueeze(0)
        c = self.c_lstm(c)
        c = self.c_ff(c)

        if self.IMITATE: a = F.log_softmax(a, dim=1)
        else: c = F.log_softmax(c, dim=1)

        return a, c

class Policy(Agent):
    def __init__(self,num_actions=2,num_chars=80,VISUALIZE=False):
        super(Policy, self).__init__()
        # replay memory
        self.memory = []; self.imitate_memory = []
        # discount rate
        self.gamma = 0.9
        # visualize
        self.visualize = VISUALIZE
        # number of environment actions
        self.num_actions = num_actions

        self.num_chars = num_chars
        # load reader network
        self.model = Model(num_actions, num_chars)
        # model optimizer
        self.opt = opt.RMSprop(self.model.parameters())
        # GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # move model to GPU
        self.model = self.model.to(self.device)

    def reset(self):
        self.model.reset_lstm()

    def save(self):
        weight_file = 'imitator_dqn' if self.model.IMITATE else 'actor_dqn'

        with open(weight_file, 'wb') as f:
            torch.save(self.model, f)

    def copy(self,imitator):
        # copy data from imitator dqn to actor
        imitator.model.reset_lstm()
        self.epsilon = imitator.epsilon
        self.model.IMITATE = imitator.model.IMITATE
        self.model.load_state_dict(imitator.model.state_dict())

    def act(self,state):
        # sample random action with probability epsilon
        if uniform(0, 1) < self.epsilon:
            return randint(0,self.num_actions-1), randint(0, self.num_chars-1)

        state = state.to(self.device)
        q_values,w = self.model(state)

        w = torch.argmax(w,dim=1)
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

    def replay(self, actor):
        # should be used to perform policy gradient update (PPO)
        batch = self.get_batch()
        action_q_values = []
        for episode in batch:
            self.reset(); actor.reset_lstm()
            for time_step in episode:
                # get experience from batch
                s,a,r,s_prime,done,correct_word = time_step

                # move data to GPU
                s, a = s.to(self.device), torch.Tensor([a]).long().to(self.device)

                # Q(s, a; theta_target)
                q_s,w = self.model(s)
                q_s_a = q_s[0,a]

                # save q values
                action_q_values.append(q_s.cpu().data.numpy()[0])

                # if the episode hasn't ended; estimate value of taking best action in s_prime
                if not done:
                    s_prime =  s_prime.to(self.device)
                    # max(Q(s', a'; theta_actor))
                    q_s_prime, w_prime = actor(s_prime)
                    q_s_a_prime = torch.max(q_s_prime)
                    # add to reward with discount factor gamma
                    r += self.gamma * q_s_a_prime

                error = r - q_s_a
                clipped_error = -1.0 * error.clamp(-1,1)

                q_s_a.backward(clipped_error, retain_graph=True)
                self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action_q_values
