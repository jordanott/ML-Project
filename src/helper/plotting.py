import matplotlib.pyplot as plt

def vs_time(data,xlabel='',ylabel='',title=''):
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
