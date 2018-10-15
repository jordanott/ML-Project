# manipulation of environment
import os
import random
from xml.etree.ElementTree import parse

from scipy.ndimage import imread

# formatting states
def crop_env():
    pass

def get_state():
    pass

# loading new environment (pages)
def load_new_env(data_dir):
    form_name = 'a01-000u.png'#random.choice(os.listdir(data_dir + 'forms/'))
    xml_name = form_name.replace('.png','.xml')

    form = imread(data_dir+'forms/'+form_name)
    xml = parse(data_dir+'xml/'+xml_name).getroot()

    return form, xml
