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
    # reset the env
    s = env.reset(); done = False
    # reset the metric monitor
    mm.reset_episode()

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
    mm.end_episode() # record metrics, increment mm.num_episodes

    # if we're acting we want to replay episodes and learn rewards
    if not IMITATE: total_loss = imitator.replay(actor.model)
    # otherwise we want to imitate the teachers actions
    else: total_loss = imitator.imitate()

    # logging: episode, losses OR reward info
    mm.log_status(total_loss)

    # set things for the next loop;
    if (mm.num_episodes+1) % SWITCH_EVERY == 0:

        # switching from imitation to acting or vice versa
        IMITATE = not IMITATE

        # set monitor and env appropriately; remove of add softmax for actions of model pred
        mm.TEACH = IMITATE; env.TEACH = IMITATE; imitator.model.IMITATE = IMITATE

        # update the actor with latest version of imitator
        actor.copy(imitator.model)

        print('Switching to ' + 'imitating' if IMITATE else 'acting')
