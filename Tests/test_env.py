import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from src.agents.dqn import DQN
from src.helper import plotting as P
from src.environment import env_manager, env_helper

EYE_DIM = 64

# define env
env = env_manager.Environment(data_dir='../data/')
# Human agent allows user to chose actions
behaviour = DQN(4)
target = DQN(4)

num_episodes = 0

while True:
    s = env.reset(); done = False
    episode = {
        'episode_len': 0,
        'reward': [],
        'history': []
    }
    behaviour.reset()

    while not done:
        # take action a
        a,w = behaviour.act(s)
        # environment returns new state, reward, done
        s_prime, r, done = env.step(a,w)
        print(num_episodes, r,done)

        # save episode info
        episode['history'].append([s,a,r,s_prime,done])
        episode['reward'].append(r); episode['episode_len'] += 1

        s = s_prime
    print('Done')
    num_episodes += 1

    target.remember(episode['history']) # s,a,r,s_prime,done)

    if num_episodes % 10 == 0:
        target.replay(behaviour.model)

        env.visualize_eyetrace()

        behaviour.copy(target.model)

        P.vs_time(episode['reward'],xlabel='Time',ylabel='Reward',title='Reward vs Time')
