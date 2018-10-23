import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from src.agents.imitator_dqn import DQN
from src.helper.monitor import MetricMonitor
from src.environment import env_manager, env_helper

IMITATE = True; SWITCH_EVERY = 10

# monitor information
mm = MetricMonitor(teach=IMITATE)
# define env
env = env_manager.Environment(data_dir='../data/',teach=IMITATE)

# Initialize agents
imitator = DQN(6); actor = DQN(6); actor.epsilon = 0

while True:
    if not IMITATE:
        # set monitor and env appropriately
        mm.TEACH = False; env.TEACH = False
        # update the actor with latest version of imitator
        actor.copy(imitator.model)

    s = env.reset(); done = False # reset the env
    mm.reset_episode() # reset the metric monitor

    while not done:
        a,w = 0,0

        # if we're not imitating, take actions
        if not IMITATE: a,w = actor.act(s)

        # environment returns new state, reward, done
        s_prime, r, done, correct_word, a = env.step(a,w)

        # save episode info
        mm.store(s,a,r,s_prime,done,correct_word)

        s = s_prime

        if (mm.num_episodes+1) % SWITCH_EVERY == 0: # visualize actions taken by the agent
            env.visualize_eyetrace(r,reward_over_time=mm.episode['reward'])

    # if the teacher results are poor, redo them
    if IMITATE and len(mm.episode['history']) < 20:
        env.num_episodes -= 1; continue

    # store the info from the episode
    imitator.remember(mm.get_history(), IMITATE)
    mm.end_episode() # record metrics, increment episode counter

    # if we're acting we want to replay episodes and learn rewards
    if not IMITATE: total_loss = imitator.replay(actor.model)
    # otherwise we want to imitate the teachers actions
    else: total_loss = imitator.imitate()

    # logging: episode, losses OR reward info
    mm.log_status(total_loss)

    if (mm.num_episodes+1) % SWITCH_EVERY == 0:
        IMITATE = not IMITATE # switching from imitation to acting or vice versa
        imitator.model.IMITATE = IMITATE # remove of add softmax for actions of model pred
        print('Switching to ' + 'imitating' if IMITATE else 'acting')
