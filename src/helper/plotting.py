import matplotlib.pyplot as plt

def vs_time(data,xlabel='',ylabel='',title='',save=True,location=''):
    plt.clf()

    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save: plt.savefig(location + ylabel + '.png')
    else: plt.show()
