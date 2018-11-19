import os
import sys
import torch
import random
import numpy as np
sys.path.append('../../')

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch import Tensor
from copy import deepcopy
from scipy.ndimage import imread
from recordclass import recordclass
from collections import namedtuple
from src.helper import plotting as P
from xml.etree.ElementTree import parse
from DataExploration.word_analysis import load_word_ids,load_char_ids


class Environment(object):
    def __init__(self,state_size=64,data_dir='../data/',M=10):
        """
        state_size (int): size of image chunk returned to the agent
        data_dir (str): location of forms and xml files
        """
        self.M = M; self.D = state_size
        self.data_dir = data_dir
        self.num_episodes = 0
        # forms to chose from 
        self.available_forms = []
        # reward for agent having eye over words
        self.word_hover_bonus = .1

        self.Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


    def new_env(self):
        self.env,self.env_words,self.lines,self.char_ids,self.whole_page_char_ids = self.gen_new_env()

    def reset(self):
        """ Generates new env, clears eye history, and sets random eye location """
        self.num_episodes += 1
        # generate new environment
        self.new_env()
        self.words = deepcopy(self.env_words)
        # x,y corresponds to the upper left coordinate of the eye box
        start_y = int(self.lines[0][0].ymin)-25; start_x = int(self.lines[0][0].xmin)-25

        EyeLocation = recordclass('EyeLocation', 'x y')
        self.eye = EyeLocation(start_x, start_y )

        # pointer for teaching
        self.P = EyeLocation(0,0)

        self.episode_count = 0
        self.done = False

        return self.format_state()

    def format_state(self):
        row, col = self.eye.y, self.eye.x

        state = self.env[row:row+self.D, col:col+self.D]
        state = state[:,:,0] / 255.
        state = state[np.newaxis,np.newaxis]
        #state += np.random.uniform(-1e-2,1e-3,size=(1,1,self.D,self.D))
        return Tensor(state)

    def visualize_eyetrace(self, r=0, show=False, reward_over_time=None):
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
        alpha = min(1, .4 if r == 0 else abs(r))

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

        if self.done:
            os.system('convert -delay 20 -loop 0 {dir}*.png {dir}env.gif'.format(dir=dir_location + 'env/'))
            os.system('convert -delay 20 -loop 0 {dir}*.png {dir}state.gif'.format(dir=dir_location + 'state/'))
            if reward_over_time is not None:
                P.vs_time(reward_over_time,xlabel='Time',ylabel='Reward',title='Reward vs Time',location=dir_location)

    # overlap between rectangles
    def overlap(self,a, b):  # returns 0 if rectangles don't intersect
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
        area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
        if (dx>=0) and (dy>=0):
            return dx*dy / float(area_a + area_b + dx*dy)
        return 0

    def load_form_and_xml(self,data_dir):
        if self.available_forms == []:
            self.available_forms = np.random.choice(os.listdir(data_dir + 'forms/'),100).tolist()
        form_file = random.choice(self.available_forms) #'a01-000u.png'
        xml_file = form_file.replace('.png','.xml')

        form = imread(data_dir+'forms/'+form_file,mode='RGB')
        xml = parse(data_dir+'xml/'+xml_file).getroot()

        return form,xml,form_file

    def eye_word_overlap(self,words, eye):
        # Find what word has most overlap with eye location
        max_coords, max_iou = None, 1e-9
        for coords in words:
            iou = self.overlap(coords, eye)
            max_iou = max(iou, max_iou)
            if max_iou == iou:
                max_coords = coords

        return max_coords, max_iou

    def nearest_word_to_point(self,words, point):
        # Find what word is closest to a given point
        point = np.array(point)
        min_coords, min_dist = None, 1e9
        for coords in words:
            c = np.array([coords.xmin, coords.ymin])
            dist = np.linalg.norm(c - point)
            min_dist = min(dist, min_dist)
            if min_dist == dist:
                min_coords = coords

        return min_coords.xmin, min_coords.ymin

    def assign_hover_bonus(self,coords):
        '''elif overlap: # reward agent for looking at words... as long as it doesn't stare too long!
            self.assign_hover_bonus(self.words[self.coords])
            r += self.word_hover_bonus * self.words[self.coords]['hover_bonus']
            self.patience -= 1'''
        # is the num times hovering over word > max times allowed
        greater_than_max_count = coords['hover_count'] > coords['max']
        # is the hover bonus > the lower bound penalty
        less_than_min_bonus = coords['hover_bonus'] > -coords['max']

        if coords['hover_count'] <= coords['max']:
            dir = 1
        elif greater_than_max_count and less_than_min_bonus:
            dir = -2
        else: dir = 0

        coords['hover_bonus'] += dir / float(coords['max'])
        coords['hover_count'] += 1

    def word_to_char_ids_swap(self,word, char_ids):
        list_char_ids = []
        for char in word:
            if type(char) == type(torch.Tensor()):
                x = int(char.cpu().numpy())
                list_char_ids.append(char_ids[x])
            else:
                list_char_ids.append(char_ids[char])
                
        return list_char_ids

    def decode(self,list_char_ids, char_ids, blank_char=0):
        processed_char_ids = []
        raw_char_ids = []

        for i in range(len(list_char_ids)):
            if i != 0 and list_char_ids[i] != list_char_ids[i-1] and list_char_ids[i] != blank_char:
                processed_char_ids.append(list_char_ids[i])
            elif i == 0 and  list_char_ids[i] != blank_char:
                processed_char_ids.append(list_char_ids[i])
            raw_char_ids.append(list_char_ids[i])
        
        return self.word_to_char_ids_swap(raw_char_ids, char_ids), self.word_to_char_ids_swap(processed_char_ids, char_ids)

    def gen_new_env(self,ds=2.0):
        # loading new environment (pages)
        form, xml,form_file = self.load_form_and_xml(self.data_dir)
        word_ids, char_ids = load_word_ids(), load_char_ids()

        lines = []
        words = {} # {coordinates : word}
        whole_page_char_ids = []
        for line in xml[1]:
            lines.append([])
            for word in line:
                try:
                  if word.tag == 'word':
                    x1,y1 = int(word[0].attrib['x']),int(word[0].attrib['y'])
                    x2,y2 = int(word[-1].attrib['x']),int(word[-1].attrib['y'])
                    w,h = int(word[-1].attrib['width']),int(word[-1].attrib['height'])

                    xmin = min(x1,x2); xmax = max(x1,x2)
                    ymin = min(y1,y2); ymax = max(y1,y2)
                    r = self.Rectangle(xmin/ds,ymin/ds,xmax/ds+w/ds,ymax/ds+h/ds)

                    # change word text to list of char ids
                    word_char_ids = self.word_to_char_ids_swap(word.attrib['text'], char_ids)

                    if r not in words:
                        words[r] = {
                            'text':word.attrib['text'],
                            'id':word_ids[word.attrib['text']],
                            'max':len(word.attrib['text']),
                            'hover_count':1,
                            'hover_bonus':1,
                            'char_ids':word_char_ids
                            }

                    lines[-1].append(r)
                    # add list of char ids plus space
                    word_char_ids.append(char_ids[' '])
                    whole_page_char_ids.extend( word_char_ids )
                except:
                  pass #print(word.attrib['text'],form_file)

        return form[::int(ds),::int(ds)], words, lines, char_ids, whole_page_char_ids
