import numpy as np
import pandas as pd
from BasisFunc import bspline_func
from SNPgvf import SNPgvf
import CreateBasis as cb
from inprod import inprod


def SimDataSNP(n, m, ytype, seed = 0):    
    np.random.seed(seed + 123)
    norder = 4
    nknots = 15
    t = np.linspace(1e-2, 1, m)
    k = norder - 1
    breaks = list(np.linspace(0, 1, nknots))
    basismat = bspline_func(t, breaks, norder)
    nbasisX = basismat.shape[1]
    coef = np.random.multivariate_normal(mean=np.zeros(nbasisX), cov=np.eye(nbasisX), size=n)
    Rawfvalue = np.dot(coef, basismat.T)
    fvalue = pd.DataFrame(Rawfvalue)

    def funcX(l):
        x = fvalue.iloc[l, :]
        diffmat = np.array([(x - i)**2 for i in range(3)])
        value = np.argmin(diffmat, axis=0)
        return value

    dataX = np.array([funcX(i) for i in range(n)])
    n_inter = 2
    funcX = SNPgvf(list(t), dataX, 'Bspline', 5, 4, Plot=False)
    fbasis1 = cb.create_bspline_basis(rangeval=[min(t), max(t)], nbasis=5, norder=4)
    fbasis2 = cb.create_bspline_basis(rangeval=[min(t), max(t)], nbasis=5, norder=4)
    n,m = dataX.shape
    funcCoef = funcX['coefs'].T
    basisint = inprod(fdobj1=fbasis1, fdobj2=fbasis2, Lfdobj1=0, Lfdobj2=0)
    def funcU(i):
        return np.dot(funcCoef[i, :], basisint)
    U = pd.DataFrame(np.array([funcU(i) for i in range(n)]).reshape(n, -1))
    z = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=n)
    dim_G = U.shape[1]
    dim_E = z.shape[1]
    INTERACTION = np.zeros(shape=(n, dim_G * dim_E))
    k = 0
    for i in range(dim_E):
        for j in range(dim_G):
            INTERACTION[:,k] = z[:,i] * U.iloc[:,j]
            k = k + 1
    coef = np.random.uniform(0.5, 0.8, size = n_inter*2+dim_E)
    coef[0] = -coef[1]
    coef[3] = -coef[2]
    y_ = np.array([np.sum(U.iloc[:,[0,-1]] * coef[0:n_inter], axis = 1) + np.sum(INTERACTION[:,[0,5]] * coef[n_inter:n_inter*2], axis = 1) + np.sum(z * coef[n_inter*2:n_inter*2+dim_E], axis = 1)])
    if ytype == 'Continuous':
        bias = np.random.rand(n).reshape(-1,1)
        y = y_.reshape(-1,1) + 0.5 * bias
        simData = {'y': y, 'z': z, 'location': list(t), 'X': dataX}
        return simData
    
    elif ytype == 'Survival':
        
        def censorData(h, n):
            U = np.random.uniform(1,3,size = n)
            MEAN = U * np.exp(h)
            TIME = np.random.exponential(np.exp(h))
            C = np.random.exponential(MEAN)
            Y_TIME = np.where(TIME > C, C, TIME)
            Y_EVENT = np.where(TIME > C, 0, 1)
            return Y_TIME.reshape(-1,1), Y_EVENT.reshape(-1,1)
        
        y = censorData(y_, n)
        y = np.array(y).reshape(2,n).T
        simData = {'y': y, 'z': z, 'location': list(t), 'X': dataX}
        return simData
    
    elif ytype == 'Binary':
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        y_class = np.where(y_ > 0.5, 1, 0)
        simData = {'y': y_class, 'z': z, 'location': list(t), 'X': dataX}
        return simData

