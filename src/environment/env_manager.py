import numpy as np
import matplotlib.pyplot as plt

from src.environment import env_helper


class Environment(object):
    def __init__(self,state_size=64,data_dir='data/'):
        """
        state_size (int): size of image chunk returned to the agent

        """
        self.data_dir = data_dir
        self.new_env()
        self.eye_movements = []

    def save_env(self):
        # save env/record actions
        pass

    def new_env(self):
        self.env,self.xml = env_helper.load_new_env(self.data_dir)

    def reset(self):
        # reset eye location
        pass

    def step(self,action):
        #
        return

    def is_terminal(self):
        # check if the state is a terminal state
        pass

    def visualize_eyetrace(self):
        # plot eye movements overlaid on image
        pass
