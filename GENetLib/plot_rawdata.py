import numpy as np
import matplotlib.pyplot as plt


'''Plot functions fitting from densely measured observations'''

def plot_rawdata(location, X, color=None, pch=4, cex=0.9, show_legend=True):
    n, m = X.shape
    type_ = 'o'
    truelengths = np.sum(~np.isnan(X))
    if truelengths == n * m:
        if color is None:
            plt.plot(location, X.T, marker=type_, markersize=pch, label='X')
        else:
            plt.plot(location, X.T, marker=type_, markersize=pch, color=color, label='X')
    else:
        location_list = [location[i][~np.isnan(X[i, :])] for i in range(n)]
        X_list = [X[i, ~np.isnan(X[i, :])] for i in range(n)]
        if color is None:
            for i in range(n):
                plt.plot(location_list[i], X_list[i], marker=type_, markersize=pch, label=f'X{i+1}')
        else:
            for i in range(n):
                plt.plot(location_list[i], X_list[i], marker=type_, markersize=pch, color=color, label=f'X{i+1}')
    plt.xlabel("Location")
    plt.ylabel("X")
    if show_legend:
        plt.legend()
    plt.show()

from GENetLib.sim_data_func import sim_data_func
seed = 123
func_survival = sim_data_func(20, 30, 'Survival', seed = seed)
location = list(func_survival['location'])
X = func_survival['X']
plot_rawdata(location, X, show_legend = False)