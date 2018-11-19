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
from Levenshtein import distance

class Reinforcer(Environment):
    def __init__(self,state_size=64,data_dir='../data/',M=10):
        super(Reinforcer, self).__init__(state_size,data_dir,M)
        self.predicted_chars = []

    def step(self, a, char):
        r = 0; self.done = False

        # add the char just predicted
        self.predicted_chars.append(char)
        print self.predicted_chars
        # decode the list of chars
        raw, pred_decoded_page = self.decode(self.predicted_chars, self.char_ids)

        print 'char', char, pred_decoded_page

        if True or len(pred_decoded_page) >= len(self.whole_page_char_ids):
            done = True
            raw, true_decode_page = self.decode(self.whole_page_char_ids, self.char_ids)

            true_decode_page = ''.join(true_decode_page)
            pred_decoded_page = ''.join(pred_decoded_page)

            print pred_decoded_page, true_decode_page
            l_dist = distance(pred_decoded_page, true_decode_page)
            r = (len(self.whole_page_char_ids) - l_dist)/ float(len(self.whole_page_char_ids))

            print 'Reward', r
        # build eye Rectangle
        #eye_rect = self.Rectangle(self.eye.x, self.eye.y, self.eye.x + self.D, self.eye.y + self.D)
        # calculate overlap between eye and words
        #self.coords, overlap = self.eye_word_overlap(self.words, eye_rect)

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
        return self.format_state(), r, self.done
