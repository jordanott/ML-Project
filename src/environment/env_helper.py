import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import namedtuple
from xml.etree.ElementTree import parse
from scipy.ndimage import imread

# formatting states
def crop_env():
    pass

def get_state():
    pass


# overlap between rectangles
def IOU(a, b):  # returns 0 if rectangles don't intersect
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

    form = imread(data_dir+'forms/'+form_file)
    xml = parse(data_dir+'xml/'+xml_file).getroot()

    return form,xml

# loading new environment (pages)
def gen_new_env(data_dir):
    form, xml = load_form_and_xml(data_dir)

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    words = {} # {coordinates : word}
    for line in xml[1]:
        for word in line:
            if word.tag == 'word':
                x1,y1 = int(word[0].attrib['x']),int(word[0].attrib['y'])
                x2,y2 = int(word[-1].attrib['x']),int(word[-1].attrib['y'])
                w,h = int(word[-1].attrib['width']),int(word[-1].attrib['height'])

                r = Rectangle(x1/2.,y1/2.,x2/2.+w/2.,y2/2.+h/2.)
                if r not in words:
                    words[r] = word.attrib['text']

    return form[::2,::2], words

def show_word_and_eye_loc(form,word,eye=None):
    # Create figure and axes
    fig,ax = plt.subplots(1)
    ax.imshow(form)
    # Create a Rectangle patch
    word_rec = patches.Rectangle((word.xmin,word.ymin),word.xmax - word.xmin,word.ymax - word.ymin,
        linewidth=1,edgecolor='g',facecolor='none')

    if eye:
        eye_rec = patches.Rectangle((eye.xmin,eye.ymin),eye.xmax - eye.xmin,eye.ymax - eye.ymin,
            linewidth=1,edgecolor='black',facecolor='none')
        ax.add_patch(eye_rec)

    # Add the patch to the Axes
    ax.add_patch(word_rec)
    plt.show()
