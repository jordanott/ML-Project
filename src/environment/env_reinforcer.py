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
    def __init__(self,state_size=64,data_dir='../data/',M=10,save_dir=''):
        super(Reinforcer, self).__init__(state_size,data_dir,M,save_dir=save_dir)
        self.predicted_chars = []
        self.predictions_file = save_dir + 'predictions.txt'
    
    def get_error_reward(self, pred_decoded_page):
        raw, true_decoded_page = self.decode(self.whole_page_char_ids, self.char_ids)

        true_decoded_page = ''.join(true_decoded_page)
        pred_decoded_page = ''.join(pred_decoded_page)

        l_dist = distance(pred_decoded_page, true_decoded_page)
        r = (len(self.whole_page_char_ids) - l_dist)/ float(len(self.whole_page_char_ids))

        with open(self.predictions_file, 'a') as f:     
            f.write('Episode:' + str(self.num_episodes)); f.write('\n')
            f.write('Predicted:\n'+''.join(pred_decoded_page)); f.write('\n')
            f.write('True:\n'+''.join(true_decoded_page)); f.write('\n')

        return max(-1, min(r, 1))

    def step(self, a, char, plot=False):
        r = 0; self.done = False

        # build eye Rectangle
        eye_rect = self.Rectangle(self.eye.x, self.eye.y, self.eye.x + self.D, self.eye.y + self.D)
        # calculate overlap between eye and words
        self.coords, overlap = self.eye_word_overlap(self.words, eye_rect)

        # add the char just predicted
        self.predicted_chars.append(char)
        
        # decode the list of chars
        raw, pred_decoded_page = self.decode(self.predicted_chars, self.char_ids)

        if len(pred_decoded_page) >= len(self.whole_page_char_ids): self.done = True

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
        
        if self.done:
            r = self.get_error_reward(pred_decoded_page)
        if plot:
            self.visualize_eyetrace(r)

        # return ( s', r, done, correct_word )
        return self.format_state(), r, self.done
