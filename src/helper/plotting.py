import numpy as np
import matplotlib.pyplot as plt

def vs_time(data,labels=None,xlabel='',ylabel='',title='',save=True,location=''):
    plt.clf()
    if labels:
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



def hist(data, labels, xlabel='', title='',save=True, location=''):
    data = np.array(data)
    n_bins = 10; plt.clf()

    plt.hist(x, n_bins, histtype='bar', label=labels)
    plt.legend()
    plt.title('bars with legend')

    if save: plt.savefig(location + xlabel + '.png')
    else: plt.show()
