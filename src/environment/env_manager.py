import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch import Tensor
from copy import deepcopy
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

    def save_env(self):
        # save env/record actions
        pass

    def new_env(self):
        self.env,self.env_words = env_helper.gen_new_env(self.data_dir)

    def reset(self):
        """ Generates new env, clears eye history, and sets random eye location """
        # generate new environment
        self.new_env() # sets self.env and self.words
        # deepcopy of env_words ~ this will be modified
        self.words = deepcopy(self.env_words)
        # keeps track of where the agent has looked
        self.eye_history = []

        # pick a random starting point the eye to look
        x,y = random.randint(0,self.env.shape[1]),random.randint(0,self.env.shape[0])
        # x,y corresponds to the upper left coordinate of the eye box
        EyeLocation = recordclass('EyeLocation', 'x y')
        self.eye = EyeLocation(x,y)

        self.count = 0

        return self.format_state()

    def format_state(self):
        row, col = self.eye.y, self.eye.x
        state = self.env[row:row+self.D, col:col+self.D]
        state = state[:,:,0] / 255.
        state = state[np.newaxis,np.newaxis]

        return Tensor(state)

    def step(self,a):
        """
        PARAM
            a (int): agent action choice 0:up, 1:right, 2:down, 3:left

        RETURNS
            s' (image): next state
            r (int): reward
            done (bool): the episode has ended
        """
        r = 0; done = False
        if a == 0: # move up
            self.eye.y -= self.M
        elif a == 1: # move right
            self.eye.x += self.M
        elif a == 2: # move down
            self.eye.y += self.M
        elif a == 3: # move left
            self.eye.x -= self.M

        self.eye_history.append(deepcopy(self.eye))

        eye_rect = env_helper.Rectangle(self.eye.x, self.eye.y, self.eye.x + self.D, self.eye.y + self.D)
        coords, overlap = env_helper.eye_word_overlap(self.words, eye_rect) # modify eye
        self.visualize_eyetrace(coords)
        self.count += 1
        return self.format_state(), r, done

    def is_terminal(self):
        # check if the state is a terminal state
        pass

    def visualize_eyetrace(self, coords=None):
        e = 1/(2. * len(self.eye_history))
        alpha = e

        fig, ax = plt.subplots()
        ax.imshow(self.env)

        for eye in self.eye_history:
            circle = plt.Circle((eye.x+self.D/2, eye.y+self.D/2), self.D/2, color='g', alpha=alpha)
            ax.add_artist(circle)
            alpha += e

        if coords:
            word_rec = patches.Rectangle((coords.xmin,coords.ymin),coords.xmax - coords.xmin,coords.ymax - coords.ymin,
                linewidth=1,edgecolor='black',facecolor='none')
            ax.add_patch(word_rec)

        plt.axis('off')
        plt.savefig('images/%03d' % self.count, bbox_inches='tight')
        plt.show()
