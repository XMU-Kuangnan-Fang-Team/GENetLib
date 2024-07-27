import numpy as np
import matplotlib.pyplot as plt


def plotRawdata(location, X, color = None, pch = 4, cex = 0.9):
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
    plt.legend()
    plt.show()


'''test
from SimDataSNP import SimDataSNP
import pandas as pd
seed = 123
snp_survival = SimDataSNP(5, 50, 'Survival', seed = seed)
y = [snp_survival['y']['time'],snp_survival['y']['event']]
z = pd.DataFrame(snp_survival['z'])
location = list(snp_survival['location'])
X = snp_survival['X']
plotRawdata(location, X)'''