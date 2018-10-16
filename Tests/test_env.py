import sys
sys.path.append('../')

import random
import numpy as np

from collections import namedtuple
from src.environment import env_manager, env_helper

EYE_DIM = 64
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

# define env
#env = env_manager.Environment(data_dir='../data/')
form, words = env_helper.gen_new_env('../data/')

# define eye
x,y = random.randint(0,form.shape[1]),random.randint(0,form.shape[0])
eye = Rectangle(x,y, x+EYE_DIM, y+EYE_DIM)

# randomly chose word to highlight
rw = random.choice(list(words.keys()))

print('Highlighted word:',words[rw])
print('Overlap with eye location:', env_helper.IOU(rw, eye))

env_helper.show_word_and_eye_loc(form,rw,eye)
