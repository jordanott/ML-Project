import random
from agent import Agent

class RandomAgent(Agent):
    """docstring for ."""
    def __init__(self, num_actions):
        super(RandomAgent,self).__init__()
        self.actions = [i for i in range(num_actions)]

    def act(self, state):
        return random.choice(self.actions)
