import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from scipy import integrate
from scipy.interpolate import BSpline, UnivariateSpline

from GENetLib.fda_func import bspline_mat


'''Example data for method scalar_ge and grid_scalar_ge'''

def sim_data_scalar(rho_G, rho_E, dim_G, dim_E, n, dim_E_Sparse = 0, ytype = 'Survival',
                  n_inter = None, linear = True, seed = 0):
    if dim_E_Sparse > dim_E:
        raise ValueError("dim_E_Sparse should be less than dim_E")
    
    def generate_continuous_data(rho, dim, n):
        cov = np.zeros(shape=(dim, dim))
        mean = np.zeros(dim)
        for i in range(dim):
            for j in range(dim):
                cov[i,j] = rho ** (abs(i-j))
        return np.random.multivariate_normal(mean = mean, cov = cov, size = n)
    
    def censor_data(h, n):
        U = np.random.uniform(1,3,size = n)
        MEAN = U * np.exp(h)
        TIME = np.random.exponential(np.exp(h))
        C = np.random.exponential(MEAN)
        Y_TIME = np.where(TIME > C, C, TIME)
        Y_EVENT = np.where(TIME > C, 0, 1)
        return {"time":Y_TIME.flatten(), "event":Y_EVENT.flatten()}
    
    np.random.seed(seed)
    X = generate_continuous_data(rho_G, dim_G, n)
    CLINICAL = generate_continuous_data(rho_E, dim_E, n)
    if dim_E_Sparse != 0:
        CLINICAL[:,dim_E-dim_E_Sparse:dim_E] = np.where(CLINICAL[:,dim_E-dim_E_Sparse:dim_E] > 0, 1, -1)
    INTERACTION = np.zeros(shape=(n, dim_G * dim_E))
    k = 0
    for i in range(dim_E):
        for j in range(dim_G):
            INTERACTION[:,k] = CLINICAL[:,i] * X[:,j]
            k = k + 1      
    if n_inter == None:
        raise ValueError("Please enter n_inter")
    else:
        pos = []
        for i in range(dim_E):
            pos += list(range(dim_G * i, dim_G * i + n_inter))
        interactionPos = np.random.choice(pos, size = n_inter, replace=False)
        
        if ytype == 'Survival':
            if linear == True:
                coef = np.random.uniform(0.5, 0.8, size = n_inter*2+dim_E)
                h = np.sum(X[:,0:n_inter] * coef[0:n_inter], axis = 1) + np.sum(INTERACTION[:,interactionPos] * coef[n_inter:n_inter*2], axis = 1) + np.sum(CLINICAL * coef[n_inter*2:n_inter*2+dim_E], axis = 1)
            elif linear == False:
                h = np.sum(np.sin(X[:,0:n_inter]), axis = 1)+np.sum(np.sin(INTERACTION[:,interactionPos]), axis = 1)+np.sum(np.sin(CLINICAL), axis = 1)
            else:
                raise ValueError("Please enter True or False")
            Y = pd.DataFrame(censor_data(h, n))
            X = StandardScaler().fit(X).transform(X)
            CLINICAL = StandardScaler().fit(CLINICAL).transform(CLINICAL)
            INTERACTION = StandardScaler().fit(INTERACTION).transform(INTERACTION)
        
        elif ytype == 'Continuous':
            coef = np.random.uniform(0.5, 0.8, size = n_inter*2+dim_E)
            bias = np.random.rand(n).reshape(-1,1)
            Y = (np.sum(X[:,0:n_inter] * coef[0:n_inter], axis = 1) + np.sum(INTERACTION[:,interactionPos] * coef[n_inter:n_inter*2], axis = 1) + np.sum(CLINICAL * coef[n_inter*2:n_inter*2+dim_E], axis = 1)).reshape(-1,1) + bias
            X = StandardScaler().fit(X).transform(X)
            CLINICAL = StandardScaler().fit(CLINICAL).transform(CLINICAL)
            INTERACTION = StandardScaler().fit(INTERACTION).transform(INTERACTION)

        elif ytype == 'Binary':
            coef = np.random.uniform(0.5, 0.8, size = n_inter*2+dim_E)
            bias = np.random.rand(n).reshape(-1,1)
            Y_ = (np.sum(X[:,0:n_inter] * coef[0:n_inter], axis = 1) + np.sum(INTERACTION[:,interactionPos] * coef[n_inter:n_inter*2], axis = 1) + np.sum(CLINICAL * coef[n_inter*2:n_inter*2+dim_E], axis = 1)).reshape(-1,1) + bias
            Y = (Y_ >= np.mean(Y_)).astype(int)
            X = StandardScaler().fit(X).transform(X)
            CLINICAL = StandardScaler().fit(CLINICAL).transform(CLINICAL)
            INTERACTION = StandardScaler().fit(INTERACTION).transform(INTERACTION)

        else:
            raise ValueError("Invalid ytype")
    return {'y':Y, 'G':X, 'E':CLINICAL, 'GE':INTERACTION,
            'interpos':interactionPos}


'''Example data for method func_ge and grid_func_ge'''

def sim_data_func(n, m, ytype, seed = 0):

    np.random.seed(seed + 123)
    norder = 4
    nknots = 20
    t = np.linspace(1e-2, 1, m)
    k = norder - 1
    breaks = list(np.linspace(0, 1, nknots))
    basismat = bspline_mat(t, breaks, norder)
    nbasisX = basismat.shape[1]
    coef = multivariate_normal.rvs(mean=np.zeros(nbasisX), cov=np.eye(nbasisX), size=n)
    Rawfvalue = np.dot(coef, basismat.T)
    fvalue = pd.DataFrame(Rawfvalue)

    def func_x(l):
        x = fvalue.iloc[l, :]
        diffmat = np.array([(x - i)**2 for i in range(3)])
        value = np.argmin(diffmat, axis=0)
        return value

    dataX = np.array([func_x(i) for i in range(n)])
    gamma = np.array([0.4, 0.8])
    np.random.seed(seed + 1234)
    z = multivariate_normal.rvs(mean=np.zeros(2), cov=np.eye(2), size=n)
    np.random.seed(seed + 12345)
    epsilon = np.random.normal(0, 0.1, n)
    region1 = t[t <= 0.3]
    region2 = t[(t > 0.3) & (t <= 0.7)]
    region3 = t[t > 0.7]
    
    Betapart1 = -36*(region1 - 0.3)**2
    Betapart2 = np.zeros(len(region2))
    Betapart3 = 36*(region3 - 0.7)**2
    
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
    
    def func_y(i):
        value = z[i, :].T @ gamma + dataX[i, :] @ basismat @ np.linalg.inv(basismat.T @ basismat) @ basisint0 + \
                 z[i, 0] * (dataX[i, :] @ basismat @ np.linalg.inv(basismat.T @ basismat) @ basisint1) + \
                 z[i, 1] * (dataX[i, :] @ basismat @ np.linalg.inv(basismat.T @ basismat) @ basisint2) + epsilon[i]
        return value
    
    if ytype == 'Survival':
        
        def censor_data(h, n):
            U = np.random.uniform(1,3,size = n)
            MEAN = U * np.exp(h)
            TIME = np.random.exponential(np.exp(h))
            C = np.random.exponential(MEAN)
            Y_TIME = np.where(TIME > C, C, TIME)
            Y_EVENT = np.where(TIME > C, 0, 1)
            return Y_TIME.reshape(-1,1), Y_EVENT.reshape(-1,1)
        
        y_ = np.array([func_y(i) for i in range(n)]).reshape(n)
        y = censor_data(y_, n)
        y = np.array(y).reshape(2,n).T
        simData = {'y': y, 'Z': z, 'location': list(t), 'X': dataX}
        return simData
    
    elif ytype == 'Continuous':
        y = np.array([func_y(i) for i in range(n)]).reshape(n)
        simData = {'y': y, 'Z': z, 'location': list(t), 'X': dataX}
        return simData
    
    elif ytype == 'Binary':
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        y_prob = np.array([func_y(i) for i in range(n)]).reshape(n)
        y_class = np.where(y_prob > 0.5, 1, 0)
        simData = {'y': y_class, 'Z': z, 'location': list(t), 'X': dataX}
        return simData
