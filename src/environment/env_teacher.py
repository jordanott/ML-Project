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

    def generate_examples(self, PER_LINE=True, plot=False):
        states_actions = [[]] if PER_LINE else []
        words = []

        if PER_LINE:
            for line in self.lines:
                words_in_line = []
                for word in line:
                    words_in_line.extend(self.env_words[word])
                words.append(words_in_line)

        while not self.done:
            # build eye Rectangle
            eye_rect = self.Rectangle(self.eye.x, self.eye.y, self.eye.x + self.D, self.eye.y + self.D)
            # calculate overlap between eye and words
            self.coords, overlap = self.eye_word_overlap(self.words, eye_rect)

            # chose action to take
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

            # store state, action
            if PER_LINE:
                states_actions[-1].append([self.format_state(), a])
            else:
                states_actions.append([self.format_state(), a])

            # move
            if a == 0: # move up
                if self.eye.y - self.M > -1: self.eye.y -= self.M/2
            elif a == 1: # move right
                if self.eye.x + self.M + self.D < self.env.shape[1]:
                    self.eye.x += self.M
                else:
                    self.a_buffer = [4]
            elif a == 2: # move down
                if self.eye.y + self.M + self.D < self.env.shape[0]: y += self.M/2
            elif a == 3: # move left
                if self.eye.x - self.M > -1: x -= self.M
            elif a == 4: # new line
                self.P.y += 1; self.P.x = 0
                if self.P.y == len(self.lines): # DONE, do something here
                    self.done = True
                elif self.eye.y + self.D < self.env.shape[0]:
                    self.eye.y = int(self.lines[self.P.y][self.P.x].ymin)-25
                    self.eye.x = int(self.lines[self.P.y][self.P.x].xmin)-25

                    if PER_LINE: states_actions.append([])


            if plot: self.visualize_eyetrace()

            self.episode_count += 1

        print len(states_actions), len(words)
        # return ( s', r, done, correct_word )
        return states_actions, words
