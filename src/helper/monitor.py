import numpy as np
from src.helper import plotting as P

class MetricMonitor(object):
    def __init__(self, num_words=13551):
        self.metrics = {
            'reward': [],
            'action_distribution': [],
            'episode_len': []
        }
        self.num_episodes = 0
        self.num_words = num_words
        self.reset_episode()

    def reset_episode(self):
        self.episode = {
            'episode_len': 0,
            'reward': [],
            'history': []
        }

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
        self.set_word_targets()

        return self.episode['history']

    def store(self, s, a, r, s_prime, done, correct_word):
        self.episode['history'].append([s,a,r,s_prime,done,correct_word])
        self.episode['reward'].append(r)
        self.episode['episode_len'] += 1

    def log_status(self):
        mean_reward = np.mean(self.metrics['reward'][-100:])
        print('Episodes: {}, Mean Reward (100): {}'.format(self.num_episodes, mean_reward))

        self.visualize_reward()

    def end_episode(self):
        self.metrics['reward'].extend(self.episode['reward'])

        self.metrics['episode_len'].append(self.episode['episode_len'])

        self.num_episodes += 1

    def visualize_reward(self):
        P.vs_time(self.metrics['reward'],xlabel='Time',ylabel='Reward',title='Reward vs Time')
