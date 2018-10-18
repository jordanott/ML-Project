import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from src.agents.dqn import DQN
from src.environment import env_manager, env_helper

EYE_DIM = 64

# define env
env = env_manager.Environment(data_dir='../data/')
# Human agent allows user to chose actions
agent = DQN(4)

s = env.reset()

while True:
    # take action a
    a = agent.act(s)
    # environment returns new state, reward, done
    s_prime, r, done = env.step(a)

    s = s_prime
