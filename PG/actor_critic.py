import torch.optim as opt
import matplotlib.pyplot as plt
import torch.nn.functional as F

from policy_actor import Actor
from policy_critic import Critic

class ActorCritic:
    def __init__(self,observation_space):

        self.actor = Actor(observation_space)
        self.critic = Critic(observation_space)

        self.actor_opt = opt.Adam(self.actor.parameters())
        self.critic_opt = opt.Adam(self.critic.parameters())

        self._actor_loss = []
        self._critic_loss = []

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
        plt.plot(self._actor_loss,label='Actor')
        plt.plot(self._critic_loss, label='Critic')

        plt.legend()
        plt.savefig('loss.png')
