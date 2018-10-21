import numpy as np
from src.helper import plotting as P

class MetricMonitor(object):
    def __init__(self, num_words=13551, teach=False):
        self.metrics = {
            'reward': [],
            'action_distribution': [],
            'episode_len': []
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
            'history': []
        }
        if self.TEACH: self.episode['history'].append([])

    def set_word_targets(self):
        # start with the last word from episode
        replacement_word = self.episode['history'][-1][-1]

        for idx in reversed(range(len(self.episode['history'])-1)):
            if replacement_word == self.num_words: continue

            current_word = self.episode['history'][idx][-1]

            if current_word == replacement_word: # if the words are the same set to READING token
                self.episode['history'][idx][-1] = self.num_words - 1

            replacement_word = current_word

    def get_history(self):
        if not self.TEACH: self.set_word_targets()

        return self.episode['history']

    def store(self, s, a, r, s_prime, done, correct_word):
        if self.TEACH:
            self.episode['history'][-1].append([s,a,r,s_prime,done,correct_word])
            if correct_word:
                self.episode['history'].append([])

        else:
            self.episode['history'].append([s,a,r,s_prime,done,correct_word])
        self.episode['reward'].append(r)
        self.episode['episode_len'] += 1

    def log_status(self, total_loss):
        if self.TEACH:
            self.imitator_loss['action_loss'].extend(total_loss['action_loss'])
            self.imitator_loss['word_loss'].extend(total_loss['word_loss'])

            data = [self.imitator_loss['action_loss'], self.imitator_loss['word_loss']]
            labels = ['Action Loss', 'Word Loss']
            P.vs_time(data, labels=labels,xlabel='Time',ylabel='Reward',title='Reward vs Time')

            print('Action loss:', np.mean(total_loss['action_loss']), 'Word loss:',np.mean(total_loss['word_loss']))

        else:
            mean_reward = np.mean(self.metrics['reward'][-100:])
            print('Episodes: {}, Mean Reward (100): {}'.format(self.num_episodes, mean_reward))

            # plot reward vs time
            P.vs_time(self.metrics['reward'],xlabel='Time',ylabel='Reward',title='Reward vs Time')

    def end_episode(self):
        self.metrics['reward'].extend(self.episode['reward'])

        self.metrics['episode_len'].append(self.episode['episode_len'])

        self.num_episodes += 1
