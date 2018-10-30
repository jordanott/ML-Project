import sys
sys.path.append('../')

import numpy as np

from src.agents.imitator_dqn import DQN
from src.helper.monitor import MetricMonitor
from src.environment import env_manager, env_helper

IMITATE = True; VISUALIZE_EVERY = 100
IMITATE_LIMIT = 10000; NET_COPY_TIME = 100

# monitor information
mm = MetricMonitor(teach=IMITATE)
# define env
env = env_manager.Environment(data_dir='../data/',teach=IMITATE)

# Initialize agents
imitator = DQN(6); actor = DQN(6)

for i in range(IMITATE_LIMIT):
    # reset the env
    s = env.reset(); done = False
    # reset the metric monitor
    mm.reset_episode()

    while not done: # loop while the episode runs
        # environment returns new state, reward, done
        s_prime, r, done, correct_word, a = env.step(0,0)

        # save episode info
        mm.store(s,a,r,s_prime,done,correct_word)

        s = s_prime

    # if the teacher results are poor, redo them
    if len(mm.episode['history']) < 30:
        env.num_episodes -= 1; continue

    # store the info from the episode
    imitator.remember(mm.get_history(), IMITATE)
    mm.end_episode() # record metrics, increment mm.num_episodes

    # imitate the teacher
    action_ctc_loss = imitator.imitate()

    # logging: episode, losses OR reward info
    mm.log_status(action_ctc_loss, 0)


# Done imitating; let the agent act on its own
IMITATE = False
# set monitor and env appropriately
mm.TEACH = IMITATE; env.TEACH = IMITATE
# remove of add softmax for actions of model pred
imitator.model.IMITATE = IMITATE


while True:
    # reset the env
    s = env.reset(); done = False
    # reset the metric monitor
    mm.reset_episode()

    while not done: # loop while the episode runs

        # take actions
        a,w = actor.act(s)

        # environment returns new state, reward, done
        s_prime, r, done, correct_word, a = env.step(a,w)

        # save episode info
        mm.store(s,a,r,s_prime,done,correct_word)

        s = s_prime

        # visualize actions taken by the agent
        if (mm.num_episodes+1) % VISUALIZE_EVERY == 0:
            env.visualize_eyetrace(r,reward_over_time=mm.episode['reward'])

    # store the info from the episode
    imitator.remember(mm.get_history(), IMITATE)
    mm.end_episode() # record metrics, increment mm.num_episodes

    # if we're acting we want to replay episodes and learn rewards
    reward_info = imitator.replay(actor.model)

    # logging: episode, losses OR reward info
    mm.log_status(reward_info, imitator.epsilon)

    if (mm.num_episodes+1) % NET_COPY_TIME == 0:
        # update the actor with latest version of imitator
        actor.copy(imitator)
        print('Actor network updated...')
