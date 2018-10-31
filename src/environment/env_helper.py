import os
import sys
import random
import numpy as np
sys.path.append('../../')
import matplotlib.pyplot as plt

from scipy.ndimage import imread
from collections import namedtuple
from xml.etree.ElementTree import parse
from DataExploration.word_analysis import load_word_ids,load_char_ids,word_to_char_ids_swap

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def get_state():
    pass

# overlap between rectangles
def overlap(a, b):  # returns 0 if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy / float(area_a + area_b + dx*dy)
    return 0

def load_form_and_xml(data_dir):
    form_file = random.choice(os.listdir(data_dir + 'forms/'))
    xml_file = form_file.replace('.png','.xml')

    form = imread(data_dir+'forms/'+form_file,mode='RGB')
    xml = parse(data_dir+'xml/'+xml_file).getroot()

    return form,xml,form_file

def eye_word_overlap(words, eye):
    """
    Find what word has most overlap with eye location
    words (dict): {coordinates: word}
    eye (recordclass): (xmin, ymin, xmax, ymax)

    returns:
        coordinates: if overlap
        None: if no overlap
    """
    max_coords, max_iou = None, 1e-9
    for coords in words:
        iou = overlap(coords, eye)
        max_iou = max(iou, max_iou)
        if max_iou == iou:
            max_coords = coords

    return max_coords, max_iou

def nearest_word_to_point(words, point):
    """
    Find what word is closest to a given point
    PARAM:
        words (dict): {coordinates: word}
        eye (list): [x, y]

    RETURN:
        coordinates
    """
    point = np.array(point)
    min_coords, min_dist = None, 1e9
    for coords in words:
        c = np.array([coords.xmin, coords.ymin])
        dist = np.linalg.norm(c - point)
        min_dist = min(dist, min_dist)
        if min_dist == dist:
            min_coords = coords

    return min_coords.xmin, min_coords.ymin

def assign_hover_bonus(coords):
    '''elif overlap: # reward agent for looking at words... as long as it doesn't stare too long!
        env_helper.assign_hover_bonus(self.words[self.coords])
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

def decode(list_char_ids, char_ids, blank_char=0):
    processed_char_ids = []

    for i in range(len(list_char_ids)):
        if i != 0 and list_char_ids[i] != list_char_ids[i-1] and list_char_ids[i] != blank_char:
            processed_char_ids.append(list_char_ids[i])

    return word_to_char_ids_swap(processed_char_ids, char_ids)

# loading new environment (pages)
def gen_new_env(data_dir,ds=2.0):
    """
    data_dir (str):
    ds (float): down sample amount
    """
    form, xml,form_file = load_form_and_xml(data_dir)
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
                r = Rectangle(xmin/ds,ymin/ds,xmax/ds+w/ds,ymax/ds+h/ds)

                # change word text to list of char ids
                word_char_ids = word_to_char_ids_swap(word.attrib['text'], char_ids)

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
