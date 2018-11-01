import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

from copy import deepcopy
from src.networks.ff import FF
from src.networks.lstm import LSTM
from warpctc_pytorch import CTCLoss
from src.networks.embedding import CNN
from random import randint,uniform,sample

class Model(nn.Module):
    def __init__(self, num_actions, num_chars):
        super(Model, self).__init__()
        # modules for action prediction
        self.a_cnn = CNN()
        self.a_lstm = LSTM(64*29*29, hidden_size=256)
        self.a_ff = FF(input_size=256,num_outputs=num_actions)

        # modules for character prediction
        self.c_cnn = CNN()
        self.c_lstm = LSTM(64*29*29, hidden_size=256)
        self.c_ff = FF(input_size=256,num_outputs=num_chars)

        self.IMITATE = True

    def reset_lstm(self):
        self.c_lstm.reset_hidden()
        self.a_lstm.reset_hidden()

    def forward(self, x):
        a = self.a_cnn(x).unsqueeze(0)
        a = self.a_lstm(a)
        a = self.a_ff(a)

        c = self.c_cnn(x).unsqueeze(0)
        c = self.c_lstm(c)
        c = self.c_ff(c)

        if self.IMITATE: a = F.log_softmax(a, dim=1)
        else: c = F.log_softmax(c, dim=1)

        return a, c

class DQN(object):
    def __init__(self,num_actions,num_chars=80,VISUALIZE=False):
        # replay memory
        self.memory = []; self.imitate_memory = []
        # discount rate
        self.gamma = 0.9
        # exploration rate
        self.epsilon = 1
        # batch size
        self.batch_size = 1
        # memory size
        self.memory_size = 2000
        # visualize
        self.visualize = VISUALIZE
        # minimum exploration
        self.epsilon_min = 0.05
        # decay
        self.epsilon_decay = 0.99
        # number of environment actions
        self.num_actions = num_actions

        self.num_chars = num_chars
        # load reader network
        self.model = Model(num_actions, num_chars)
        # model optimizer
        self.opt = opt.RMSprop(self.model.parameters())
        # GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # move model to GPU
        self.model = self.model.to(self.device)

    def reset(self):
        self.model.reset_lstm()

    def save(self):
        weight_file = 'imitator_dqn' if self.model.IMITATE else 'actor_dqn'
        
        with open(weight_file, 'wb') as f:
            torch.save(self.model, f)

    def copy(self,imitator):
        # copy data from imitator dqn to actor
        imitator.model.reset_lstm()
        self.epsilon = imitator.epsilon
        self.model.IMITATE = imitator.model.IMITATE
        self.model.load_state_dict(imitator.model.state_dict())

    def act(self,state):
        # sample random action with probability epsilon
        if uniform(0, 1) < self.epsilon:
            return randint(0,self.num_actions-1), randint(0, self.num_chars-1)

        state = state.to(self.device)
        q_values,w = self.model(state)

        w = torch.argmax(w,dim=1)
        a = torch.argmax(q_values,dim=1).cpu().data.numpy()[0]

        return a,w

    def remember(self,episode):
        if len(self.memory) == self.memory_size:
            del self.memory[0]

        self.memory.append(episode)

    def get_batch(self):
        # if replay memory is big enough take random sample
        if len(self.memory) > self.batch_size:
            return sample(self.memory,self.batch_size)

        return self.memory

    def replay(self, actor):
        batch = self.get_batch()
        action_q_values = []
        for episode in batch:
            self.reset(); actor.reset_lstm()
            for time_step in episode:
                # get experience from batch
                s,a,r,s_prime,done,correct_word = time_step

                # move data to GPU
                s, a = s.to(self.device), torch.Tensor([a]).long().to(self.device)

                # Q(s, a; theta_target)
                q_s,w = self.model(s)
                q_s_a = q_s[0,a]

                # save q values
                action_q_values.append(q_s.cpu().data.numpy()[0])

                # if the episode hasn't ended; estimate value of taking best action in s_prime
                if not done:
                    s_prime =  s_prime.to(self.device)
                    # max(Q(s', a'; theta_actor))
                    q_s_prime, w_prime = actor(s_prime)
                    q_s_a_prime = torch.max(q_s_prime)
                    # add to reward with discount factor gamma
                    r += self.gamma * q_s_a_prime

                error = r - q_s_a
                clipped_error = -1.0 * error.clamp(-1,1)

                q_s_a.backward(clipped_error, retain_graph=True)
                self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action_q_values

    def imitate(self, states_actions, words):
        CTC = CTCLoss()
        total_action_loss = 0
        action_loss_history = []; char_predictions = []
        self.reset()

        for s, a in states_actions:
            # move data to GPU
            s, a = s.to(self.device), torch.Tensor([a]).long().to(self.device)
            # network predicted action and char
            action_pred,char_pred = self.model(s)

            # save character predicted for ctc loss
            char_predictions.append(char_pred)

            # action loss
            action_loss = F.nll_loss(action_pred, a)
            # record action loss
            action_loss_history.append(action_loss.item()); total_action_loss += action_loss

        # Must be an IntTensor!
        words = torch.Tensor(words).int()
        # seq x batch x alphabet size
        char_predictions = torch.stack(char_predictions)

        pred_len = torch.IntTensor([char_predictions.shape[0]])
        word_len = torch.IntTensor([words.shape[0]])

        ctc_loss = CTC(char_predictions, words, pred_len, word_len)

        # calculate total loss
        loss = ctc_loss.to(self.device) + total_action_loss

        self.opt.zero_grad()
        loss.backward(retain_graph=True)
        self.opt.step()

        greedy = char_predictions.squeeze().argmax(dim=1)

        return {'word_loss': ctc_loss.item(), 'action_loss': np.mean(action_loss_history)}, greedy


    '''def imitate(self):
        total_loss = {'action_loss': [], 'word_loss':[]}

        for word_focus in self.imitate_memory:
            self.reset() # reset hidden states
            action_loss_history = []

            for saccade in word_focus:
                action_loss, word_loss = 0,0
                # get experience from saccade
                s,a,r,s_prime,done,correct_word = saccade

                # move data to GPU
                s, a = s.to(self.device), torch.Tensor([a]).long().to(self.device)

                # network predicted action and word
                action_pred,word_pred = self.model(s)

                # action loss
                action_loss = F.nll_loss(action_pred, a)

                # record action loss
                action_loss_history.append(action_loss.item())

                if correct_word: # if its time to predict a word
                    word_pred = word_pred.to(self.device)
                    correct_word = torch.Tensor([correct_word]).long().to(self.device)

                    word_loss = F.nll_loss(word_pred, correct_word)
                    # record losses
                    total_loss['action_loss'].append(np.mean(action_loss_history))
                    total_loss['word_loss'].append(word_loss.item())
                    action_loss_history = []

                # total loss: combination of action and word losses
                loss = action_loss + word_loss

                self.opt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt.step()

        return total_loss
        '''
