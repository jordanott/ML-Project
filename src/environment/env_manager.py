import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from recordclass import recordclass
from src.environment import env_helper

class Environment(object):
    def __init__(self,state_size=64,data_dir='data/',M=10):
        """
        state_size (int): size of image chunk returned to the agent
        data_dir (str): location of forms and xml files
        """
        self.M = M
        self.D = state_size
        self.data_dir = data_dir
        # generate new environment
        self.new_env() # sets self.env and self.words
        # keeps track of where the agent has looked
        self.eye_movements = []

        # pick a random starting point the eye to look
        x,y = random.randint(0,self.env.shape[1]),random.randint(0,self.env.shape[0])
        # x,y corresponds to the upper left coordinate of the eye box
        EyeLocation = recordclass('EyeLocation', 'x y')
        self.eye = EyeLocation(x,y)

    def save_env(self):
        # save env/record actions
        pass

    def new_env(self):
        self.env,self.words = env_helper.gen_new_env(self.data_dir)

    def reset(self):
        # reset eye location
        pass

    def format_state(self):
        row, col = self.eye.y, self.eye.x
        return self.env[row:row+self.D, col:col+self.D]

    def step(self,a):
        if a == 0: # move up
            self.eye.y -= self.M
        elif a == 1: # move right
            self.eye.x += self.M
        elif a == 2: # move down
            self.eye.y += self.M
        elif a == 3: # move left
            self.eye.x -= self.M

        self.eye_movements.append(copy.deepcopy(self.eye))

        eye_rect = env_helper.Rectangle(self.eye.x, self.eye.y, self.eye.x + self.D, self.eye.y + self.D)
        coords, overlap = env_helper.eye_word_overlap(self.words, eye_rect) # modify eye
        self.visualize_eyetrace(coords)

        return self.format_state()

    def is_terminal(self):
        # check if the state is a terminal state
        pass

    def visualize_eyetrace(self, coords=None):
        e = 1/(2. * len(self.eye_movements))
        alpha = e

        fig, ax = plt.subplots()
        ax.imshow(self.env)

        for eye in self.eye_movements:
            circle = plt.Circle((eye.x+self.D/2, eye.y+self.D/2), self.D/2, color='g', alpha=alpha)
            ax.add_artist(circle)
            alpha += e

        if coords:
            word_rec = patches.Rectangle((coords.xmin,coords.ymin),coords.xmax - coords.xmin,coords.ymax - coords.ymin,
                linewidth=1,edgecolor='black',facecolor='none')
            ax.add_patch(word_rec)

        plt.show()
