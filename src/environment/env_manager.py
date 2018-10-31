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

    def save_env(self):
        # save env/record actions
        pass

    def new_env(self):
        self.env,self.env_words,self.lines,self.char_ids,self.whole_page_char_ids = env_helper.gen_new_env(self.data_dir)

    def reset(self):
        """ Generates new env, clears eye history, and sets random eye location """
        self.num_episodes += 1
        # generate new environment
        self.new_env() # sets self.env and self.words
        # deepcopy of env_words ~ this will be modified
        self.words = deepcopy(self.env_words)
        # set patience
        self.patience = 100
        # coordinates of current word
        self.coords = None

        # x,y corresponds to the upper left coordinate of the eye box
        start_y = int(self.lines[0][0].ymin)-25; start_x = int(self.lines[0][0].xmin)-25

        EyeLocation = recordclass('EyeLocation', 'x y')
        self.eye = EyeLocation(start_x, start_y )

        # pointer for teaching
        self.P = EyeLocation(0,0)

        self.predicted_chars = []
        self.episode_count = 0

        return self.format_state()

    def format_state(self):
        row, col = self.eye.y, self.eye.x

        state = self.env[row:row+self.D, col:col+self.D]
        state = state[:,:,0] / 255.
        state = state[np.newaxis,np.newaxis]

        return Tensor(state)

    def step(self, a, char):
        r = 0; self.done = False; correct_word = None

        # add the char just predicted
        self.predicted_chars.append(char)
        # decode the list of chars
        decoded_word = self.decode(self.predicted_chars)

        # check if the decoded word is correct; assign rewards
        if decoded_word == correct_word:
            r = 1; self.patience = 75; del self.words[self.coords]
        elif len(decoded_word) >= len(correct_word): r = -1

        # build eye Rectangle
        eye_rect = env_helper.Rectangle(self.eye.x, self.eye.y, self.eye.x + self.D, self.eye.y + self.D)
        # calculate overlap between eye and words
        self.coords, overlap = env_helper.eye_word_overlap(self.words, eye_rect)

        if a == 0: # move up
            if self.eye.y - self.M > -1: self.eye.y -= self.M/2
        elif a == 1: # move right
            if self.eye.x + self.M + self.D < self.env.shape[1]: self.eye.x += self.M
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

        self.patience -= 1; self.episode_count += 1

        # if patience has run out the episode ends
        if self.patience == 0: self.done = True

        # return ( s', r, done, correct_word )
        return self.format_state(), r, self.done, correct_word, a

    def visualize_eyetrace(self, r, show=False, reward_over_time=None):
        dir_location = 'images/%05d/' % self.num_episodes
        if not os.path.exists(dir_location): # mkdir
            os.mkdir(dir_location); os.mkdir(dir_location + 'env'); os.mkdir(dir_location + 'state')

        ################################ ENV ##################################
        plt.clf(); fig, ax = plt.subplots()
        ax.imshow(self.env)

        # assign color based on sign of reward
        if r > 0: c = 'g'
        elif r == 0: c = 'b'
        else: c = 'r'
        # oppacity based on magnitude of reward
        alpha = min(1, .1 if r == 0 else abs(r))

        circle = plt.Circle((self.eye.x+self.D/2, self.eye.y+self.D/2), self.D/2, color=c, alpha=alpha)
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

        if self.patience == 1 or self.done:
            os.system('convert -delay 20 -loop 0 {dir}*.png {dir}env.gif'.format(dir=dir_location + 'env/'))
            os.system('convert -delay 20 -loop 0 {dir}*.png {dir}state.gif'.format(dir=dir_location + 'state/'))
            if reward_over_time is not None:
                P.vs_time(reward_over_time,xlabel='Time',ylabel='Reward',title='Reward vs Time',location=dir_location)
