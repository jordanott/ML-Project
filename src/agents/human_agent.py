import random
import matplotlib.pyplot as plt
from agent import Agent

class HumanAgent(Agent):
    """docstring for ."""
    def __init__(self,plot=False):
        super(HumanAgent,self).__init__()
        self.PLOT = plot
        
    def act(self, state):
        if self.PLOT:
            plt.imshow(state)
            plt.show()

        a = int(input('Action: '))
        return a
