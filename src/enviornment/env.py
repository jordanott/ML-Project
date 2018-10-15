import numpy as np
import matplotlib.pyplot as plt

from env_helper import *


class Environment(object):
    def __init__(self,state_size=64,):
        """
        state_size (int): size of image chunk returned to the agent

        """
        self.new_env()

    def save_env(self):
        # save env/record actions

    def new_env(self):
        self.env = env_helper.load_new_env()

    def reset(self):
        # reset eye location

    def step(self,action):
        #
        return

    def is_terminal(self):
        # check if the state is a terminal state

    def visualize_eyetrace(self):
        # plot eye movements overlaid on image
