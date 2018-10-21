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
    def __init__(self,state_size=64,data_dir='data/',M=10,teach=False):
        """
        state_size (int): size of image chunk returned to the agent
        data_dir (str): location of forms and xml files
        """
        self.M = M; self.D = state_size
        self.data_dir = data_dir
        self.num_episodes = 0
        # reward for agent having eye over words
        self.word_hover_bonus = .1

        self.TEACH = teach

    def save_env(self):
        # save env/record actions
        pass

    def new_env(self):
        self.env,self.env_words,self.lines = env_helper.gen_new_env(self.data_dir)

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
        # pick a random starting point the eye to look
        #self.start_x,self.start_y = env_helper.nearest_word_to_point(self.words, [0,0])
        #self.start_x,self.start_y = int(self.start_x)-25,int(self.start_y)-25

        # x,y corresponds to the upper left coordinate of the eye box
        start_y = int(self.lines[0][0].ymin)-25; start_x = int(self.lines[0][0].xmin)-25

        EyeLocation = recordclass('EyeLocation', 'x y')
        self.eye = EyeLocation(start_x, start_y )

        # pointer for teaching
        self.P = EyeLocation(0,0)

        # action buffer
        self.a_buffer = []

        self.episode_count = 0

        return self.format_state()

    def format_state(self):
        row, col = self.eye.y, self.eye.x

        state = self.env[row:row+self.D, col:col+self.D]
        state = state[:,:,0] / 255.
        state = state[np.newaxis,np.newaxis]

        return Tensor(state)

    def step(self, a, word):
        r = -.3; done = False; correct_word = None
        # build eye Rectangle
        eye_rect = env_helper.Rectangle(self.eye.x, self.eye.y, self.eye.x + self.D, self.eye.y + self.D)
        # calculate overlap between eye and words
        self.coords, overlap = env_helper.eye_word_overlap(self.words, eye_rect)

        if self.TEACH:
            a = 1 if np.random.uniform() < .7 else np.random.choice([0,2,3],p=[.4,.4,.2])
            if self.a_buffer:
                a = self.a_buffer.pop()
            elif self.P.x == len(self.lines[self.P.y]):
                self.a_buffer = [4] + [np.random.choice([0,1,2,3],p=[.2,.5,.2,.1]) for _ in range(3)]
            else:
                current_word = self.lines[self.P.y][self.P.x]

                # CHECK ~ classification
                eye_rect = env_helper.Rectangle(self.eye.x + self.M, self.eye.y, self.eye.x + self.M + self.D, self.eye.y + self.D)
                coords, overlap = env_helper.eye_word_overlap(self.words, eye_rect)
                TIME_TO_CLASSIFY = coords != current_word
                # CHECK ~ going down
                eye_rect = env_helper.Rectangle(self.eye.x, self.eye.y + self.M, self.eye.x + self.D, self.eye.y + self.M + self.D)
                GOING_DOWN = env_helper.overlap(current_word, eye_rect) == 1e-9
                # CHECK ~ going up
                eye_rect = env_helper.Rectangle(self.eye.x, self.eye.y - self.M, self.eye.x + self.D, self.eye.y - self.M + self.D)
                GOING_UP = env_helper.overlap(current_word, eye_rect) == 1e-9

                eye_rect = env_helper.Rectangle(self.eye.x, self.eye.y, self.eye.x + self.D, self.eye.y + self.D)
                OVERLAP = env_helper.overlap(current_word, eye_rect) > 1e-9

                #dist_away = self.eye.y + self.D/2.0 - (current_word.ymax + current_word.ymin) / 2.0
                SMALL_WORD = current_word.xmax - current_word.xmin < self.M

                END_LINE_CLASSIFY = OVERLAP and self.eye.x + self.M + self.D > self.env.shape[1]

                if TIME_TO_CLASSIFY and SMALL_WORD and OVERLAP:
                    self.a_buffer = [5,1]
                elif (TIME_TO_CLASSIFY and OVERLAP) or END_LINE_CLASSIFY:
                    a = 5
                elif GOING_UP and a == 0:
                    a = 2
                elif GOING_DOWN and a == 2:
                    a = 0
                elif not TIME_TO_CLASSIFY and END_LINE_CLASSIFY:
                    a = 5

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
                print('Done with document...')
            elif self.eye.y + self.D < self.env.shape[0]:
                self.eye.y = int(self.lines[self.P.y][self.P.x].ymin)-25
                self.eye.x = int(self.lines[self.P.y][self.P.x].xmin)-25
        elif a == 5: # classify
            r = -1; self.P.x += 1

            if self.coords is not None:

                if self.TEACH: word = self.words[self.coords]['id']
                correct_word = self.words[self.coords]['id']

                if self.words[self.coords]['id'] == word: # if the agent correctly predicts the word
                    r = 1; del self.words[self.coords] # remove word if predicted correctly
                    self.patience = 75
                '''elif overlap: # reward agent for looking at words... as long as it doesn't stare too long!
                    env_helper.assign_hover_bonus(self.words[self.coords])
                    r += self.word_hover_bonus * self.words[self.coords]['hover_bonus']
                    self.patience -= 1'''
            else:
                correct_word = len(self.env_words) + 1

        self.patience -= 1; self.episode_count += 1

        if self.patience == 0:
            done = True
        # return ( s', r, done, correct_word )
        return self.format_state(), r, done, correct_word

    def visualize_eyetrace(self, r, show=False, reward_over_time=None):
        dir_location = 'images/%05d/' % self.num_episodes

        if not os.path.exists(dir_location): # mkdir
            os.mkdir(dir_location); os.mkdir(dir_location + 'env'); os.mkdir(dir_location + 'state')

        #######################################################################
        ################################ ENV ##################################
        #######################################################################
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
        #######################################################################
        ################################ STATE ################################
        #######################################################################
        plt.clf(); fig, ax = plt.subplots()

        state = self.format_state().squeeze().numpy() # get current state
        ax.imshow(np.stack([state,state,state],axis=2)); plt.axis('off')
        plt.savefig( dir_location + 'state/' + '%05d' % self.episode_count, bbox_inches='tight')

        if show: plt.show()

        if self.patience == 1:
            os.system('convert -delay 20 -loop 0 {dir}*.png {dir}env.gif'.format(dir=dir_location + 'env/'))
            os.system('convert -delay 20 -loop 0 {dir}*.png {dir}state.gif'.format(dir=dir_location + 'state/'))
            if reward_over_time:
                P.vs_time(reward_over_time,xlabel='Time',ylabel='Reward',title='Reward vs Time',location=dir_location)
