import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from copy import deepcopy
from src.networks.ff import FF
from src.networks.embedding import CNN
from random import randint,uniform,sample

class DQN(object):
    def __init__(self,num_actions,VISUALIZE=False):
        # replay memory
        self.memory = []
        # discount rate
        self.gamma = 0.8
        # exploration rate
        self.epsilon = 1
        # batch size
        self.batch_size = 256
        # memory size
        self.memory_size = 2000
        # visualize
        self.visualize = VISUALIZE
        # minimum exploration
        self.epsilon_min = 0.05
        # decay
        self.epsilon_decay = 0.99999
        # number of environment actions
        self.action_size = num_actions
        # load reader network
        self.model = nn.Sequential(
            CNN(),
            FF(64*29*29, num_actions)
        )

    def copy(self,model):
        self.model = deepcopy(model)

    def remember(self,state,action,reward,s_prime,done):
        if len(self.memory) == self.memory_size:
            del self.memory[0]
        # store experience
        self.memory.append([state,action,reward,s_prime,done])

    def act(self,state):
        # sample random action with probability epsilon
        if uniform(0, 1) < self.epsilon:
            return randint(0,self.action_size-1)

        q_values = self.model(state)
        a = torch.argmax(q_values,dim=1).data.numpy()[0]

        return a

    def get_batch(self):
        # if replay memory is big enough take random sample
        if len(self.memory) > self.batch_size:
            return sample(self.memory,self.batch_size)

        return self.memory

    '''def replay(self):
        batch = self.get_batch()
        x,y = [],[]
        for i in range(len(batch)):
            # get experience from batch
            state,action,reward,s_prime,done = batch[i]
            target = reward
            # if the round hasn't ended; estimate value of taking best action in s_prime
            if not done:
                p_reward = np.amax(self.model.predict(s_prime)[0])
                # add to reward with discount factor gamma
                target = reward + self.gamma * p_reward
            # predict greedy reward of current state
            target_f = self.model.predict(state)
            # set the observed reward of action previously taken
            target_f[0][action] = target
            # add to training set for network
            x.append(state.reshape(state.shape[1],state.shape[2],state.shape[3],1))
            y.append(target_f[0])
        self.model.fit(np.array(x), np.array(y), epochs=2, verbose=0)
        self.memory = []
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay'''
