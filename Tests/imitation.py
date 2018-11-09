import sys
sys.path.append('../')

import numpy as np

from src.agents.dqn import DQN
from src.helper.monitor import MetricMonitor
from src.environment import env_reinforcer, env_teacher

PER_LINE = True; VISUALIZE_EVERY = 100; IMITATE_LIMIT = 100000; NET_COPY_TIME = 100

# monitor information
mm = MetricMonitor(teach=True)
# define env
teacher = env_teacher.Teacher()
reinforcer = env_reinforcer.Reinforcer()

# Initialize imitator agent
imitator = DQN(5, PER_LINE=PER_LINE)

for i in range(IMITATE_LIMIT):
    # reset the env
    s = teacher.reset()
    # reset the metric monitor
    mm.reset_episode()

    states_actions, words = teacher.generate_examples(PER_LINE=PER_LINE)
    # imitate the teacher
    action_ctc_loss, greedy_pred = imitator.imitate(states_actions, words)

    true_decode = teacher.word_to_char_ids_swap(words, teacher.char_ids)
    pred_decode = teacher.decode(greedy_pred, teacher.char_ids)

    mm.end_episode() # record metrics, increment mm.num_episodes

    # logging: episode, losses OR reward info
    mm.log_status(action_ctc_loss, 0)

    if (i + 1) % NET_COPY_TIME == 0:
        print 'Saving imitator model'
        #imitator.save()
        print 'Predicted:\n', ''.join(pred_decode)
        print
        print 'True:\n', ''.join(true_decode)

# set monitor and env appropriately
mm.TEACH = False; reinforcer.TEACH = False

# the target network will learn a Q function
target = imitator
# remove of add softmax for actions of model pred
target.model.IMITATE = False

# the actor takes actions which the target network will learn from
actor = DQN(5); actor.copy(target)

while True:
    # reset the env
    s = reinforcer.reset(); done = False
    # reset the metric monitor
    mm.reset_episode()

    while not done: # loop while the episode runs

        # take actions
        a,w = actor.act(s)

        # environment returns new state, reward, done
        s_prime, r, done, correct_word, a = reinforcer.step(a,w)

        # save episode info
        mm.store(s,a,r,s_prime,done,correct_word)

        s = s_prime

        # visualize actions taken by the agent
        if (mm.num_episodes+1) % VISUALIZE_EVERY == 0:
            reinforcer.visualize_eyetrace(r,reward_over_time=mm.episode['reward'])

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
