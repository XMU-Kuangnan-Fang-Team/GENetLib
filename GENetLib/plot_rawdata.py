import numpy as np
import matplotlib.pyplot as plt


'''Plot functions fitting from densely measured observations'''

def plot_rawdata(location, X, color = None, pch = 4, cex = 0.9):
    n, m = X.shape
    type_ = 'o'
    truelengths = np.sum(~np.isnan(X))
    if truelengths == n * m:
        if color is None:
            plt.plot(location, X.T, marker=type_, markersize=pch, label='X')
        else:
            plt.plot(location, X.T, marker=type_, markersize=pch, color=color, label='X')
    plt.xlabel("Location")
    plt.ylabel("X")
    plt.legend()
    plt.show()

