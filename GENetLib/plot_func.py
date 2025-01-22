from GENetLib.plot_fd import plot_fd


'''Plot functional objects regardless of whether y inputs or not'''

def plot_func(x, y = None, xlab = None, ylab = None):

    tofunc = x
    if y == None:
        plot_fd(tofunc, xlab, ylab)
    else:
        plot_fd(tofunc, y, xlab, ylab)

