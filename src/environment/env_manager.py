import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch import Tensor
from copy import deepcopy
from recordclass import recordclass
from src.helper import plotting as P
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
        self.num_episodes = 0
        # reward for agent having eye over words
        self.word_hover_bonus = .1

    def save_env(self):
        # save env/record actions
        pass

    def new_env(self):
        self.env,self.env_words = env_helper.gen_new_env(self.data_dir)

    def reset(self):
        """ Generates new env, clears eye history, and sets random eye location """
        self.num_episodes += 1
        # generate new environment
        self.new_env() # sets self.env and self.words
        # deepcopy of env_words ~ this will be modified
        self.words = deepcopy(self.env_words)
        # keeps track of where the agent has looked
        self.eye_history = []
        # set patience
        self.patience = 25
        # coordinates of current word
        self.coords = None
        # pick a random starting point the eye to look
        self.start_x,self.start_y = env_helper.nearest_word_to_point(self.words, [0,0])
        self.start_x,self.start_y = int(self.start_x)-25,int(self.start_y)-25

        #random.randint(0,self.env.shape[1]-self.D),random.randint(0,self.env.shape[0]-self.D)
        # x,y corresponds to the upper left coordinate of the eye box
        EyeLocation = recordclass('EyeLocation', 'x y')
        self.eye = EyeLocation(self.start_x,self.start_y )

        self.episode_count = 0

        return self.format_state()

    def format_state(self):
        row, col = self.eye.y, self.eye.x

        state = self.env[row:row+self.D, col:col+self.D]
        state = state[:,:,0] / 255.
        state = state[np.newaxis,np.newaxis]

        return Tensor(state)

    def step(self, a, word):
        """
        PARAM
            a (int): agent action choice 0:up, 1:right, 2:down, 3:left
            word (int): ID of word predicted by the agent

        RETURNS
            s' (image): next state
            r (int): reward = word_bonus + word_hover_bonus
            done (bool): the episode has ended
        """
        r = -.3; done = False

        if a == 0: # move up
            if self.eye.y - self.M > -1: self.eye.y -= self.M
        elif a == 1: # move right
            if self.eye.x + self.M + self.D < self.env.shape[1]: self.eye.x += self.M
        elif a == 2: # move down
            if self.eye.y + self.M + self.D < self.env.shape[0]: self.eye.y += self.M
        elif a == 3: # move left
            if self.eye.x - self.M > -1: self.eye.x -= self.M
        elif a == 4: # new line
            if self.eye.y + self.D < self.env.shape[0]:
                self.eye.y += self.D
                self.eye.x = self.start_x

        # record where the eye was looking
        self.eye_history.append(deepcopy(self.eye))

        # build eye Rectangle
        eye_rect = env_helper.Rectangle(self.eye.x, self.eye.y, self.eye.x + self.D, self.eye.y + self.D)

        # calculate overlap between eye and words
        self.coords, overlap = env_helper.eye_word_overlap(self.words, eye_rect)

        if self.coords is not None:
            correct_word = self.words[self.coords]['id']
            if self.words[self.coords]['id'] == word: # if the agent correctly predicts the word
                r = 1; del self.words[self.coords] # remove word if predicted correctly

            elif overlap: # reward agent for looking at words... as long as it doesn't stare too long!
                env_helper.assign_hover_bonus(self.words[self.coords])
                r += self.word_hover_bonus * self.words[self.coords]['hover_bonus']
                self.patience -= 1
        else:
            correct_word = len(self.env_words) + 1
            self.patience -= 1

        self.episode_count += 1

        if self.patience == 0:
            done = True
        # return ( s', r, done, correct_word )
        return self.format_state(), r, done, correct_word

    def visualize_eyetrace(self, r, show=False, reward_over_time=None):
        dir_location = 'images/%05d/' % self.num_episodes
        if not os.path.exists(dir_location):
            os.mkdir(dir_location)

        plt.clf()
        fig, ax = plt.subplots()
        ax.imshow(self.env)

        if r > 0: c = 'g'
        elif r == 0: c = 'b'
        else: c = 'r'

        alpha = min(1, .1 if r == 0 else abs(r))

        circle = plt.Circle((self.eye.x+self.D/2, self.eye.y+self.D/2), self.D/2, color=c, alpha=alpha)
        ax.add_artist(circle)

        if self.coords:
            word_rec = patches.Rectangle((self.coords.xmin,self.coords.ymin),self.coords.xmax - self.coords.xmin,self.coords.ymax - self.coords.ymin,
                linewidth=1,edgecolor='black',facecolor='none')
            ax.add_patch(word_rec)

        plt.axis('off')
        plt.savefig('images/%05d/%05d' % (self.num_episodes,self.episode_count), bbox_inches='tight')

        if show: plt.show()

        if self.patience == 1:
            os.system('convert -delay 20 -loop 0 {dir}0*.png {dir}summary.gif'.format(dir=dir_location))
            if reward_over_time:
                P.vs_time(reward_over_time,xlabel='Time',ylabel='Reward',title='Reward vs Time',location=dir_location)
