import os
import random
import numpy as np
import matplotlib
matplotlib.use('agg')
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
        self.M = M; self.D = state_size
        self.data_dir = data_dir
        self.num_episodes = 0
        # reward for agent having eye over words
        self.word_hover_bonus = .1

    def reset(self):
        """ Generates new env, clears eye history, and sets random eye location """
        self.num_episodes += 1
        # generate new environment
        self.env,self.words,self.lines,self.char_ids,self.whole_page_char_ids = env_helper.gen_new_env(self.data_dir)

        # x,y corresponds to the upper left coordinate of the eye box
        start_y = int(self.lines[0][0].ymin)-25; start_x = int(self.lines[0][0].xmin)-25

        EyeLocation = recordclass('EyeLocation', 'x y')
        self.eye = EyeLocation(start_x, start_y )

        # pointer for teaching
        self.P = EyeLocation(0,0)

        self.episode_count = 0
        self.done = False
        self.a_buffer = []

    def format_state(self):
        row, col = self.eye.y, self.eye.x

        state = self.env[row:row+self.D, col:col+self.D]
        state = state[:,:,0] / 255.
        state = state[np.newaxis,np.newaxis]

        return Tensor(state)

    def generate_examples(self, plot=False):
        states_actions = []

        while not self.done:
            # build eye Rectangle
            eye_rect = env_helper.Rectangle(self.eye.x, self.eye.y, self.eye.x + self.D, self.eye.y + self.D)
            # calculate overlap between eye and words
            self.coords, overlap = env_helper.eye_word_overlap(self.words, eye_rect)

            a = 1 if np.random.uniform() < .5 else np.random.choice([0,2,3],p=[.4,.4,.2])
            if self.a_buffer:
                a = self.a_buffer.pop()
            elif self.P.x == len(self.lines[self.P.y]):
                self.a_buffer = [4] + [np.random.choice([0,1,2,3],p=[.2,.5,.2,.1]) for _ in range(3)]
            else:
                ymin = max(self.lines[self.P.y][0].ymin, self.lines[self.P.y][-1].ymin)
                ymax = min(self.lines[self.P.y][0].ymax, self.lines[self.P.y][-1].ymax)

                # if we want to go up and going up would put us out of reach from ymin
                if a == 0 and self.eye.y - self.M/2. + self.D < ymin:
                    a = 2
                elif a == 2 and self.eye.y + self.M/2. > ymax:
                    a = 0

            if a == 0: # move up
                if self.eye.y - self.M > -1: self.eye.y -= self.M/2
            elif a == 1: # move right
                if self.eye.x + self.M + self.D < self.env.shape[1]:
                    self.eye.x += self.M
                else:
                    self.a_buffer = [4]
            elif a == 2: # move down
                if self.eye.y + self.M + self.D < self.env.shape[0]: self.eye.y += self.M/2
            elif a == 3: # move left
                if self.eye.x - self.M > -1: self.eye.x -= self.M
            elif a == 4: # new line
                self.P.y += 1; self.P.x = 0
                if self.P.y == len(self.lines): # DONE, do something here
                    self.done = True
                elif self.eye.y + self.D < self.env.shape[0]:
                    self.eye.y = int(self.lines[self.P.y][self.P.x].ymin)-25
                    self.eye.x = int(self.lines[self.P.y][self.P.x].xmin)-25

            states_actions.append([self.format_state(), a])

            if plot:
                self.visualize_eyetrace()

            self.episode_count += 1
        # return ( s', r, done, correct_word )
        return states_actions, self.whole_page_char_ids

    def visualize_eyetrace(self, show=False, reward_over_time=None):
        dir_location = 'images/%05d/' % self.num_episodes
        if not os.path.exists(dir_location): # mkdir
            os.mkdir(dir_location); os.mkdir(dir_location + 'env'); os.mkdir(dir_location + 'state')

        ################################ ENV ##################################
        plt.clf(); fig, ax = plt.subplots()
        ax.imshow(self.env)

        circle = plt.Circle((self.eye.x+self.D/2, self.eye.y+self.D/2), self.D/2, alpha=.5)
        ax.add_artist(circle)

        if self.coords:
            word_rec = patches.Rectangle((self.coords.xmin,self.coords.ymin),self.coords.xmax - self.coords.xmin,self.coords.ymax - self.coords.ymin,
                linewidth=1,edgecolor='y',facecolor='none')
            ax.add_patch(word_rec)

        plt.axis('off')
        plt.savefig( dir_location + 'env/' + '%05d' % self.episode_count, bbox_inches='tight')

        if show: plt.show()

        ################################ STATE ################################
        plt.clf(); fig, ax = plt.subplots()

        state = self.format_state().squeeze().numpy() # get current state
        ax.imshow(np.stack([state,state,state],axis=2)); plt.axis('off')
        plt.savefig( dir_location + 'state/' + '%05d' % self.episode_count, bbox_inches='tight')

        if show: plt.show()

        if self.done:
            os.system('convert -delay 20 -loop 0 {dir}*.png {dir}env.gif'.format(dir=dir_location + 'env/'))
            os.system('convert -delay 20 -loop 0 {dir}*.png {dir}state.gif'.format(dir=dir_location + 'state/'))
            if reward_over_time is not None:
                P.vs_time(reward_over_time,xlabel='Time',ylabel='Reward',title='Reward vs Time',location=dir_location)
