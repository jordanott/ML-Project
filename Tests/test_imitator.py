import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from src.agents.imitator_dqn import DQN
from src.helper.monitor import MetricMonitor
from src.environment import env_manager, env_helper

# monitor information
mm = MetricMonitor(teach=True)

# define env
env = env_manager.Environment(data_dir='../data/',teach=True)

# Initialize agents
imitator = DQN(6)

while True:
    s = env.reset(); done = False

    mm.reset_episode()

    while not done:
        # take action a
        #a,w = behaviour.act(s)
        a,w = 0,0
        # environment returns new state, reward, done
        s_prime, r, done, correct_word = env.step(a,w)

        # save episode info
        mm.store(s,a,r,s_prime,done,correct_word)

        s = s_prime

        if (mm.num_episodes+1) % 100 == 0:
            env.visualize_eyetrace(r,reward_over_time=mm.episode['reward'])

    imitator.remember(mm.get_history())
    mm.end_episode()

    if mm.num_episodes % 10 == 0:
        total_loss = imitator.replay()

        mm.log_status(total_loss)
