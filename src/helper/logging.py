import sys
import time

def update(message):
    sys.stdout.write('\r{}'.format(message))
    sys.stdout.flush()

def new_line():
    sys.stdout.write('\n')
