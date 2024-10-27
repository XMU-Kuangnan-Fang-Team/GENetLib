import numpy as np
import matplotlib.pyplot as plt

from GENetLib.eval_basis_fd import eval_fd


'''Plot functional objects'''

def plot_fd(x, y = None, xlab = None, ylab = None):
    fdobj = x
    coef = fdobj['coefs']
    coefd = coef.shape
    ndim = len(coefd)
    nbasis = coefd[0]
    nx = np.max([501, 10 * nbasis + 1])
    nrep = coefd[1]
    basisobj = fdobj['basis']
    rangex = basisobj['rangeval']
    if y == None:
        y = nx
    if y >= 1:
        y = list(np.linspace(rangex[0], rangex[1], num=int(y)))
    else:
        raise ValueError("'y' is a single number less than one.")
    xlim = rangex
    fdmat = eval_fd(y, fdobj, 0)
    rangey = [np.min(fdmat), np.max(fdmat)]
    ylim = rangey
    plt.figure()
    for irep in range(nrep):
        plt.plot(y, fdmat[:, irep])
    plt.axhline(0, linestyle='--', color='black')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    
