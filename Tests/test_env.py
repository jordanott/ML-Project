import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from src.agents.dqn import DQN
from src.helper.monitor import MetricMonitor
from src.environment import env_manager, env_helper

# monitor information
mm = MetricMonitor()

# define env
env = env_manager.Environment(data_dir='../data/')

# Initialize agents
behaviour = DQN(4); target = DQN(4)
behaviour.copy(target.model)

while True:
    s = env.reset(); done = False

    mm.reset_episode()
    behaviour.reset()

    while not done:
        # take action a
        a,w = behaviour.act(s)
        # environment returns new state, reward, done
        s_prime, r, done, correct_word = env.step(a,w)

        # save episode info
        mm.store(s,a,r,s_prime,done,correct_word)

        s = s_prime

        if (mm.num_episodes + 1) % 100 == 0:
            env.visualize_eyetrace(r,reward_over_time=mm.episode['reward'])

    target.remember(mm.get_history())
    mm.end_episode()

    if mm.num_episodes % 10 == 0:
        target.replay(behaviour.model)

        behaviour.copy(target.model)

        mm.log_status()
