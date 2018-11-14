import numpy as np
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.nn.functional as F

from policy_actor import Actor
from policy_critic import Critic

class ActorCritic:
    def __init__(self,observation_space):

        self.actor = Actor(observation_space)
        self.critic = Critic(observation_space)

        self.actor_opt = opt.Adam(self.actor.parameters(), lr=.001)
        self.critic_opt = opt.Adam(self.critic.parameters(), lr=0.1)

        self._actor_loss = []
        self._critic_loss = []

        self.loss = {'actor':[], 'critic':[]}

    def act(self, state):

        return self.actor(state)

    def value_estimate(self, state):
        self._value_estimate = self.critic(state)
        return self._value_estimate

    def update(self, s, td_target, td_error, action):
        self.actor_loss(action, td_error)
        self.critic_loss(s, td_target)

    def actor_loss(self, action, target):
        loss = -self.actor.dist.log_prob(action) * target
        # Add cross entropy cost to encourage exploration
        loss -= 1e-1 * self.actor.dist.entropy()

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

        self._actor_loss.append(loss.item())

    def critic_loss(self, s, target):
        loss = F.mse_loss(self.critic(s), target)

        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

        self._critic_loss.append(loss.item())

    def plot(self):
        plt.clf()

        self.loss['actor'].append(np.mean(self._actor_loss))
        self.loss['critic'].append(np.mean(self._critic_loss))

        self._actor_loss = []; self._critic_loss = []

        plt.plot(self.loss['actor'],label='Actor')
        plt.plot(self.loss['critic'], label='Critic')

        plt.legend()
        plt.savefig('loss.png')


        plt.clf()

        plt.plot(self.actor.history['mu'],label='Mu')
        plt.plot(self.actor.history['sigma'], label='Sigma')

        plt.legend()
        plt.savefig('Dist.png')
