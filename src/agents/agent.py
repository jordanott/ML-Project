import torch
from warpctc_pytorch import CTCLoss

class Agent(object):
    """Agent super class"""
    def __init__(self):
        pass

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

        ctc_loss = CTC(char_predictions, words, pred_len, word_len).to(self.device)

        if ctc_loss.item() == float('inf') or ctc_loss.item() == -float('inf'):
            print("WARNING: received an inf loss, setting loss value to 0")
            ctc_loss = 0

        # calculate total loss
        loss = ctc_loss + total_action_loss

        self.opt.zero_grad()
        loss.backward(retain_graph=True)

        # Norm cutoff to prevent explosion of gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 400)
        self.opt.step()

        greedy = char_predictions.squeeze().argmax(dim=1)

        return {'word_loss': ctc_loss.item(), 'action_loss': np.mean(action_loss_history)}, greedy


'''def imitate(self):
    ## word level imitation

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
