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
behaviour = DQN(4)
target = DQN(4)


while True:

    s = env.reset(); done = False
    episode_count = 1

    behaviour.reset()
    while not done:
        # take action a
        a,w = behaviour.act(s)
        # environment returns new state, reward, done
        s_prime, r, done = env.step(a,w)

        target.remember(s,a,r,s_prime,done)

        target.replay(behaviour.model)

        s = s_prime

        if episode_count % 100 == 0:
            env.visualize_eyetrace()

            behaviour.copy(target.model)
        episode_count += 1
