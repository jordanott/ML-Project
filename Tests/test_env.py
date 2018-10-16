import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from src.agents.random_agent import RandomAgent
from src.environment import env_manager, env_helper

EYE_DIM = 64

# define env
env = env_manager.Environment(data_dir='../data/')
#form, words = env_helper.gen_new_env('../data/')
agent = RandomAgent(10)


#s = env.reset()
s = None
while True:
    a = agent.act(s)

    # user input actions
    a = int(input('>'))

    s_prime = env.step(a)

    s = s_prime
    #plt.imshow(s_prime)
    #plt.show()
