import os
import torch
import random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch import Tensor
from copy import deepcopy
from environment import Environment
from recordclass import recordclass
from src.helper import plotting as P

class Teacher(Environment):
    def __init__(self,state_size=64,data_dir='../data/',M=10):
        super(Teacher, self).__init__(state_size,data_dir,M)

        self.a_buffer = []

    def update_eye(self,x,y):
        self.eye.x = x
        self.eye.y = y

    def update_pointer(self, x,y):
        self.P.x = x
        self.P.y = y

    def generate_examples(self, PER_WORD=True, plot=False):
        states_actions = [[]] if PER_WORD else []
        words = [] if PER_WORD else self.whole_page_char_ids

        while not self.done:

            x, y = self.eye.x, self.eye.y
            px, py = self.P.x, self.P.y

            print 'Word', py, px
            # build eye Rectangle
            eye_rect = self.Rectangle(x, y, x + self.D, y + self.D)
            # calculate overlap between eye and words
            self.coords, overlap = self.eye_word_overlap(self.words, eye_rect)

            a = 1 if np.random.uniform() < .5 else np.random.choice([0,2,3],p=[.4,.4,.2])
            if self.a_buffer:
                a = self.a_buffer.pop()
            elif self.P.x == len(self.lines[self.P.y]):
                self.a_buffer = 4 #[4] + [np.random.choice([0,1,2,3],p=[.2,.5,.2,.1]) for _ in range(3)]
            else:
                ymin = max(self.lines[self.P.y][0].ymin, self.lines[self.P.y][-1].ymin)
                ymax = min(self.lines[self.P.y][0].ymax, self.lines[self.P.y][-1].ymax)

                # if we want to go up and going up would put us out of reach from ymin
                if a == 0 and y - self.M/2. + self.D < ymin:
                    a = 2
                elif a == 2 and y + self.M/2. > ymax:
                    a = 0

            if a == 0: # move up
                if y - self.M > -1: y -= self.M/2
            elif a == 1: # move right
                if x + self.M + self.D < self.env.shape[1]:
                    x += self.M
                else:
                    self.a_buffer = [4]
            elif a == 2: # move down
                if y + self.M + self.D < self.env.shape[0]: y += self.M/2
            elif a == 3: # move left
                if x - self.M > -1: x -= self.M
            elif a == 4: # new line
                py += 1; px = 0
                if self.P.y == len(self.lines): # DONE, do something here
                    self.done = True
                elif y + self.D < self.env.shape[0]:
                    print y,x
                    print self.P.y, self.P.x
                    y = int(self.lines[py][px].ymin)-25
                    x = int(self.lines[py][px].xmin)-25

            if PER_WORD:
                states_actions[-1].append([self.format_state(), a])
            else:
                states_actions.append([self.format_state(), a])

            # record when we finished looking over a word
            if PER_WORD and (self.eye.x > self.lines[self.P.y][self.P.x].xmax or a == 4):
                px += 1
                word_char_ids = self.env_words[self.lines[self.P.y][self.P.x]]['char_ids']
                words.append(list(word_char_ids))
                states_actions.append([])

            if plot: self.visualize_eyetrace()

            self.episode_count += 1
            self.update_eye(x,y); self.update_pointer(px, py)
        print len(states_actions), len(words)
        # return ( s', r, done, correct_word )
        return states_actions, words
