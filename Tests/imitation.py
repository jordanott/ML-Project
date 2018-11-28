import os
import sys
sys.path.append('../')
import datetime
import numpy as np

from src.agents.dqn import DQN
from src.helper.monitor import MetricMonitor
from src.environment import env_reinforcer, env_teacher

# imitate or act, size of observation, prefix for directory name
IMITATE = True; GLIMPSE = 64; PREFIX='{}_'.format(GLIMPSE)

# ctc per line, visualize every x steps, how many steps to imitate, how often targets are set
PER_LINE = True; VISUALIZE_EVERY = 100; IMITATE_LIMIT = 50000; NET_COPY_TIME = 5000

# creating directories
CUR_DIR = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y").replace(' ','-') +'/'
if not os.path.exists(CUR_DIR):
    os.mkdir(CUR_DIR)
if not os.path.exists(CUR_DIR+'images'):
    os.mkdir(CUR_DIR+'images')

# monitor information
mm = MetricMonitor(teach=True,save_dir=CUR_DIR)
# define env
teacher = env_teacher.Teacher(save_dir=CUR_DIR,state_size=GLIMPSE)
reinforcer = env_reinforcer.Reinforcer(save_dir=CUR_DIR,state_size=GLIMPSE)

# Initialize imitator agent
imitator = DQN(5, PER_LINE=PER_LINE,save_dir=CUR_DIR)

if IMITATE:
    for i in range(IMITATE_LIMIT):
        # reset the env
        s = teacher.reset()
        # reset the metric monitor
        mm.reset_episode()

        states_actions, words = teacher.generate_examples(PER_LINE=PER_LINE)
        
        # imitate the teacher
        action_ctc_loss, greedy_pred = imitator.imitate(states_actions, words)

        # flatten word list
        if PER_LINE: words = [w for line in words for w in line]
 
        true_decode = teacher.word_to_char_ids_swap(words, teacher.char_ids)
        raw, pred_decode = teacher.decode(greedy_pred, teacher.char_ids)

        mm.end_episode() # record metrics, increment mm.num_episodes

        # logging: episode, losses OR reward info
        mm.log_status(action_ctc_loss, 0)

        if (i + 1) % 100 == 0:
            print 'Saving imitator model'
            imitator.save(prefix=PREFIX)
            print 'Predicted:\n', ''.join(pred_decode)
            print
            print 'True:\n', ''.join(true_decode)
else:
    imitator.load_weights(char_net_weights='imitator_char_net') # , act_net_weights='imitator_act_net')
    print 'Weights loaded'

# set monitor and env appropriately
mm.TEACH = False; reinforcer.TEACH = False

# the target network will learn a Q function
target = imitator
# remove of add softmax for actions of model pred
target.act_net.IMITATE = False

# the actor takes actions which the target network will learn from
actor = DQN(5, save_dir=CUR_DIR); actor.copy(target); plot=False

while True:
    # reset the env
    s = reinforcer.reset(); done = False
    # reset the metric monitor
    mm.reset_episode()

    while not done: # loop while the episode runs

        # take actions
        a,w = actor.act(s)

        # environment returns new state, reward, done
        s_prime, r, done = reinforcer.step(a,w,plot=plot)

        # save episode info
        mm.store(s,a,r,s_prime,done)

        s = s_prime
    plot = False
    # store the info from the episode
    actor.remember(mm.get_history())
    mm.end_episode() # record metrics, increment mm.num_episodes

    # if we're acting we want to replay episodes and learn rewards
    reward_info = actor.replay(target)

    # logging: episode, losses OR reward info
    mm.log_status(reward_info, actor.epsilon)

    if (mm.num_episodes+1) % NET_COPY_TIME == 0:
        # update the actor with latest version of target
        target.copy(actor); actor.save()
        print('Actor network saved and updated...')
        plot = True
