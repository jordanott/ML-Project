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
from environment import Environment
from src.helper import plotting as P

class Reinforcer(Environment):
    def __init__(self,state_size=64,data_dir='../data/',M=10):
        super(Reinforcer, self).__init__(state_size,data_dir,M)
        self.predicted_chars = []

    def step(self, a, char):
        # TODO: set correct word ; error here from it being None
        r = 0; self.done = False; correct_word = None

        # add the char just predicted
        self.predicted_chars.append(char)
        # decode the list of chars
        decoded_word = self.decode(self.predicted_chars, self.char_ids)

        # check if the decoded word is correct; assign rewards
        if decoded_word == correct_word:
            r = 1; self.patience = 75; del self.words[self.coords]
        elif len(decoded_word) >= len(correct_word): r = -1

        # build eye Rectangle
        eye_rect = self.Rectangle(self.eye.x, self.eye.y, self.eye.x + self.D, self.eye.y + self.D)
        # calculate overlap between eye and words
        self.coords, overlap = self.eye_word_overlap(self.words, eye_rect)

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
