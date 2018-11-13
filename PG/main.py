import gym
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing

from actor_critic import ActorCritic
from sklearn.kernel_approximation import RBFSampler

gamma = .9

env = gym.envs.make('MountainCarContinuous-v0')
observation_space = env.observation_space.sample()




observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

def featurize_state(state):
    """
    Returns the featurized representation for a state.
    """
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]

ac = ActorCritic(featurize_state(observation_space).shape[0])

while True:
    done = False
    s = env.reset()
    reward = []
    while not done:
        s = featurize_state(s)
        env.render()
        action = ac.act(s)

        s_prime, r, done, _ = env.step([action])

        reward.append(r)

        value_next = ac.value_estimate(featurize_state(s_prime)).detach()
        td_target = r + gamma * value_next
        current_value = ac.value_estimate(s).detach()
        td_error = td_target - current_value

        ac.update(s, td_target, td_error, action)

        s = s_prime

        if len(reward) % 100 == 0:
            print 'Avg reward:', np.mean(reward)
            reward = []
            ac.plot()
