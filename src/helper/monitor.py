import numpy as np
from src.helper import plotting as P

class MetricMonitor(object):
    def __init__(self, num_words=13551, teach=False):
        self.metrics = {
            'reward': [],
            'action_distribution': [],
            'episode_len': [],
            'num_words_seen': []
        }
        self.imitator_loss = {
            'action_loss': [],
            'word_loss': []
        }
        self.num_episodes = 0
        self.num_words = num_words

        self.TEACH = teach
        self.reset_episode()

    def reset_episode(self):
        self.episode = {
            'episode_len': 0,
            'reward': [],
            'history': [],
            'action_distribution': [0,0,0,0,0,0]
        }
        if self.TEACH: self.episode['history'].append([])

    def set_word_targets(self): # this will be used when training model from scratch (without imitation learning)
        # start with the last word from episode
        replacement_word = self.episode['history'][-1][-1]

        for idx in reversed(range(len(self.episode['history'])-1)):
            if replacement_word == self.num_words: continue

            current_word = self.episode['history'][idx][-1]

            if current_word == replacement_word: # if the words are the same set to READING token
                self.episode['history'][idx][-1] = self.num_words - 1

            replacement_word = current_word

    def get_history(self):
        #if not self.TEACH: self.set_word_targets()
        return self.episode['history']

    def store(self, s, a, r, s_prime, done, correct_word):
        if self.TEACH:
            self.episode['history'][-1].append([s,a,r,s_prime,done,correct_word])
            if correct_word:
                self.episode['history'].append([])
        else:
            self.episode['history'].append([s,a,r,s_prime,done,correct_word])

        self.episode['action_distribution'][a] += 1
        self.episode['reward'].append(r)
        self.episode['episode_len'] += 1

    def log_status(self, total_loss, epsilon=None):
        action_labels = ['Up','Right','Down','Left','New Line','Classify']

        if self.TEACH:
            # store losses from imitation training
            self.imitator_loss['action_loss'].extend(total_loss['action_loss'])
            self.imitator_loss['word_loss'].extend(total_loss['word_loss'])

            data = [self.imitator_loss['action_loss'], self.imitator_loss['word_loss']]
            losses = ['Action Loss', 'Word Loss']
            # plot losses from imitation training
            P.vs_time(data, labels=losses,xlabel='Time',ylabel='Loss',title='Loss vs Time')

            # plot number of words seen in training
            P.vs_time(self.metrics['num_words_seen'],xlabel='Time',ylabel='Words Seen',title='Words Seen vs Time')

            print('Episodes: {} Action loss: {} Word loss: {}'.format(
                self.num_episodes, np.mean(total_loss['action_loss']),np.mean(total_loss['word_loss'])))
        else:
            # mean reward from acting
            mean_reward = np.mean(self.episode['reward'])
            max_reward = np.max(self.episode['reward'])

            # plot reward for acting
            P.vs_time(self.metrics['reward'],xlabel='Time',ylabel='Reward',title='Reward vs Time')

            # plot histogram of action q values for the episode
            P.hist(total_loss, action_labels, xlabel='Q Values', title='Action Q Values',save=True, location='images/'+str(self.num_episodes))

            print('Episodes: {}, Mean Reward: {}, Max: {}, Epsilon: {}'.format(
                self.num_episodes, mean_reward, max_reward, epsilon))

        actions = np.array(self.metrics['action_distribution'])
        action_distribution = [actions[:,i] for i in range(6)]
        # plot distribution of actions taken for episode
        P.vs_time(action_distribution, labels=action_labels,xlabel='Time',ylabel='Action Probabilities',title='Action Probabilities vs Time')

    def end_episode(self):
        # store episode (reward, episode length, and num words seen)
        self.metrics['reward'].extend(self.episode['reward'])
        self.metrics['episode_len'].append(self.episode['episode_len'])
        self.metrics['num_words_seen'].append(len(self.episode['history']))
        # calc action distribution
        x = np.array(self.episode['action_distribution'])
        self.metrics['action_distribution'].append(x / x.sum().astype(np.float))
        # increment num episodes
        self.num_episodes += 1
