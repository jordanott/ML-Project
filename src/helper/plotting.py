import matplotlib.pyplot as plt

def vs_time(data,labels=None,xlabel='',ylabel='',title='',save=True,location=''):
    plt.clf()
    if type(data) == list:
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
