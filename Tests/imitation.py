import sys
sys.path.append('../')

import numpy as np

from src.agents.imitator_dqn import DQN
from src.helper.monitor import MetricMonitor
from src.environment import env_manager, env_helper, env_teacher

VISUALIZE_EVERY = 100; IMITATE_LIMIT = 10000; NET_COPY_TIME = 100

# monitor information
mm = MetricMonitor(teach=True)
# define env
teacher = env_teacher.Environment(data_dir='../data/')
env = env_manager.Environment(data_dir='../data/')

# Initialize imitator agent
imitator = DQN(6)

for i in range(IMITATE_LIMIT):
    # reset the env
    s = teacher.reset()
    # reset the metric monitor
    mm.reset_episode()

    states_actions, words = teacher.generate_examples()
    # imitate the teacher
    action_ctc_loss, greedy_pred = imitator.imitate(states_actions, words)

    pred_decode = env_helper.decode(greedy_pred, teacher.char_ids)
    true_decode = env_helper.word_to_char_ids_swap(words, teacher.char_ids)

    print 'Predicted:', ''.join(pred_decode)
    print 'True:', ''.join(true_decode)

    mm.end_episode() # record metrics, increment mm.num_episodes

    # logging: episode, losses OR reward info
    mm.log_status(action_ctc_loss, 0)

    if (i + 1) % NET_COPY_TIME == 0:
        print 'Saving imitator model'
        imitator.save()

# set monitor and env appropriately
mm.TEACH = False; env.TEACH = False

# the target network will learn a Q function
target = imitator
# remove of add softmax for actions of model pred
target.model.IMITATE = False

# the actor takes actions which the target network will learn from
actor = DQN(6); actor.copy(target)

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
    target.remember(mm.get_history())
    mm.end_episode() # record metrics, increment mm.num_episodes

    # if we're acting we want to replay episodes and learn rewards
    reward_info = target.replay(actor.model)

    # logging: episode, losses OR reward info
    mm.log_status(reward_info, target.epsilon)

    if (mm.num_episodes+1) % NET_COPY_TIME == 0:
        # update the actor with latest version of target
        actor.copy(target); actor.save()
        print('Actor network saved and updated...')
