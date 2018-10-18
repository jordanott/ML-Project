import os
import random
import matplotlib.pyplot as plt

from collections import namedtuple
from xml.etree.ElementTree import parse
from scipy.ndimage import imread

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
    form_file = 'a01-000u.png'#random.choice(os.listdir(data_dir + 'forms/'))
    xml_file = form_file.replace('.png','.xml')

    form = imread(data_dir+'forms/'+form_file,mode='RGB')
    xml = parse(data_dir+'xml/'+xml_file).getroot()

    return form,xml

def eye_word_overlap(words, eye):
    """
    Find what word
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

# loading new environment (pages)
def gen_new_env(data_dir,ds=2.0):
    """
    data_dir (str):
    ds (float): down sample amount
    """
    form, xml = load_form_and_xml(data_dir)

    words = {} # {coordinates : word}
    for line in xml[1]:
        for word in line:
            if word.tag == 'word':
                x1,y1 = int(word[0].attrib['x']),int(word[0].attrib['y'])
                x2,y2 = int(word[-1].attrib['x']),int(word[-1].attrib['y'])
                w,h = int(word[-1].attrib['width']),int(word[-1].attrib['height'])

                r = Rectangle(x1/ds,y1/ds,x2/ds+w/ds,y2/ds+h/ds)
                if r not in words:
                    words[r] = word.attrib['text']

    return form[::int(ds),::int(ds)], words
