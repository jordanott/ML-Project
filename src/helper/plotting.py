import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt

def vs_time(data,labels=None,xlabel='',ylabel='',title='',save=True,location=''):
    plt.clf()
    if labels is not None:
        for d,l in zip(data,labels):
            plt.plot(d, label=l)
        plt.legend()
    else:
        plt.plot(data)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save: plt.savefig(location + ylabel + '.png')
    else: plt.show()

def state_word_prediction(state, word_prob, words, n=5, save=True, location=''):
    plt.clf(); fig=plt.figure()

    ax=fig.add_subplot(1,2,1)
    ax.axis('off')
    ax.imshow(state)

    ax=fig.add_subplot(1,2,2)
    # indexes of n largest values
    n_largest_idx = word_prob.argsort()[-n:][::-1]
    # probability values of n largest
    n_largest_prob = word_prob[n_largest_idx]
    # word labels of n largest probable words
    word_labels = [words[i] for i in n_largest_idx]

    ax.bar(np.arange(n), n_largest_prob, tick_label=word_labels)
    ax.set_xlabel('Word Probabilities')

    if save: plt.savefig(location + ylabel + '.png')
    else: plt.show()

def hist(data, labels, xlabel='', title='',save=True, location=''):
    data = np.array(data)
    n_bins = 10; plt.clf()

    plt.hist(data, n_bins, histtype='bar', label=labels)
    plt.xlabel(xlabel)
    plt.legend()
    plt.title(title)

    if save: plt.savefig(location + xlabel + '.png')
    else: plt.show()
