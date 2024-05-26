import numpy as np
import pandas as pd
from BasisFunc import bspline_func
from CreateBasis import create_bspline_basis
from scipy.stats import multivariate_normal
from scipy import integrate
from scipy.interpolate import BSpline, UnivariateSpline
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
    fbasisX = create_bspline_basis(rangeval = [0,1], breaks = breaks, norder = norder)
    nbasisX = basismat.shape[1]
    coef = multivariate_normal.rvs(mean=np.zeros(nbasisX), cov=np.eye(nbasisX), size=n)
    Rawfvalue = np.dot(coef, basismat.T)
    fvalue = pd.DataFrame(Rawfvalue)

    def funcX(l):
        x = fvalue.iloc[l, :]
        diffmat = np.array([(x - i)**2 for i in range(3)])
        value = np.argmin(diffmat, axis=0)
        return value

    dataX = np.array([funcX(i) for i in range(n)])
    gamma = np.array([0.4, 0.8])
    np.random.seed(seed + 1234)
    z = multivariate_normal.rvs(mean=np.zeros(2), cov=np.eye(2), size=n)
    np.random.seed(seed + 12345)
    epsilon = np.random.normal(0, 0.1, n)
    region1 = t[t <= 0.3]
    region2 = t[(t > 0.3) & (t <= 0.7)]
    region3 = t[t > 0.7]
    Betapart1 = 3*(region1 - 1) * np.sin(2 * np.pi * (region1 + 0.2))
    Betapart2 = np.zeros(len(region2))
    Betapart3 = - 3 * region3 * np.sin(2 * np.pi * (region3 - 0.2))
    beta0value = np.concatenate((Betapart1, Betapart2, Betapart3))
    beta1value = np.concatenate((Betapart1, Betapart2, np.zeros(len(Betapart3))))
    beta2value = np.concatenate((np.zeros(len(Betapart1)), Betapart2, Betapart3))
    beta0fd = UnivariateSpline(t, beta0value, k=2, s=5e-4)
    beta1fd = UnivariateSpline(t, beta1value, k=2, s=5e-4)
    beta2fd = UnivariateSpline(t, beta2value, k=2, s=5e-4)
    rangeval = (0, 1)
    knots = np.concatenate(([rangeval[0]]*(norder-1), breaks, [rangeval[1]]*(norder-1)))
    coefficients = np.eye(len(breaks) + norder - 2)
    fbasisX = [BSpline(knots, coefficients[i], norder - 1) for i in range(len(breaks) + norder - 2)]
    basisint0 = np.zeros(len(breaks) + k - 1)
    basisint1 = np.zeros(len(breaks) + k - 1)
    basisint2 = np.zeros(len(breaks) + k - 1)
    for i in range(len(breaks) + k - 1):
        basisint0[i] = integrate.quad(lambda x: fbasisX[i](x) * beta0fd(x), rangeval[0], rangeval[1])[0]
        basisint1[i] = integrate.quad(lambda x: fbasisX[i](x) * beta1fd(x), rangeval[0], rangeval[1])[0]
        basisint2[i] = integrate.quad(lambda x: fbasisX[i](x) * beta2fd(x), rangeval[0], rangeval[1])[0]
    
    def funcY(i):
        value = z[i, :].T @ gamma + dataX[i, :] @ basismat @ np.linalg.inv(basismat.T @ basismat) @ basisint0 + \
                 z[i, 0] * (dataX[i, :] @ basismat @ np.linalg.inv(basismat.T @ basismat) @ basisint1) + \
                 z[i, 1] * (dataX[i, :] @ basismat @ np.linalg.inv(basismat.T @ basismat) @ basisint2) + epsilon[i]
        return value
    if ytype == 'Survival':
        def censorData(h, n):
            U = np.random.uniform(1,3,size = n)
            MEAN = U * np.exp(h)
            TIME = np.random.exponential(np.exp(h))
            C = np.random.exponential(MEAN)
            print(min(TIME),max(TIME))
            print(min(C),max(C))
            Y_TIME = np.where(TIME > C, C, TIME)
            Y_EVENT = np.where(TIME > C, 0, 1)
            return Y_TIME.reshape(-1,1), Y_EVENT.reshape(-1,1)
        pos = []
        n_inter = 2
        funcX = SNPgvf(list(t), dataX, 'Bspline', 5, 4, Plot=False)
        fbasis1 = cb.create_bspline_basis(rangeval=[min(t), max(t)], nbasis=5, norder=4)
        fbasis2 = cb.create_bspline_basis(rangeval=[min(t), max(t)], nbasis=20, norder=4)
        n,m = dataX.shape
        funcCoef = funcX['coefs'].T
        basisint = inprod(fdobj1=fbasis1, fdobj2=fbasis2, Lfdobj1=0, Lfdobj2=0)
        def funcU(i):
            return np.dot(funcCoef[i, :], basisint)
        U = pd.DataFrame(np.array([funcU(i) for i in range(n)]).reshape(n, -1))
        dim_G = U.shape[1]
        dim_E = z.shape[1]
        INTERACTION = np.zeros(shape=(n, dim_G * dim_E))
        k = 0
        for i in range(dim_E):
            for j in range(dim_G):
                INTERACTION[:,k] = z[:,i] * U.iloc[:,j]
                k = k + 1
        for i in range(dim_E):
            pos += list(range(dim_G * i, dim_G * i + n_inter))
        interactionPos = np.random.choice(pos, size = n_inter, replace=False)
        coef = np.random.uniform(0.5, 0.8, size = n_inter*2+dim_E)
        h = np.sum(U.iloc[:,0:n_inter] * coef[0:n_inter], axis = 1) + np.sum(INTERACTION[:,interactionPos] * coef[n_inter:n_inter*2], axis = 1) + np.sum(z * coef[n_inter*2:n_inter*2+dim_E], axis = 1)
        y = censorData(h, n)
        simData = {'y': y, 'z': z, 'location': t, 'X': dataX}
        return simData
    elif ytype == 'Continuous':
        y = np.array([funcY(i) for i in range(n)]).reshape(n)
        simData = {'y': y, 'z': z, 'location': t, 'X': dataX}
        return simData
    elif ytype == 'Binary':
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        y_prob = np.array([funcY(i) for i in range(n)]).reshape(n)
        y_class = np.where(y_prob > 0.5, 1, 0)
        simData = {'y': y_class, 'z': z, 'location': t, 'X': dataX}
        return simData

