import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

from GENetLib.dense_to_func import dense_to_func
import GENetLib.create_basis as cb
from GENetLib.inprod import inprod
from GENetLib.grid_scalar_ge import grid_scalar_ge
from GENetLib.eval_basis_fd import eval_fd
from GENetLib.fd import fd


'''Grid search for func_ge'''

def grid_func_ge(y, z, location, X, ytype, btype, num_hidden_layers, nodes_hidden_layer,
                 Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs,
                 nbasis1, params1, t = None, Bsplines = 20, norder1 = 4, 
                 model = None, split_type = 0, ratio = [7, 3], plot_res = True, plot_beta = True):
    # When X is functional input
    if type(X) == dict:
        fbasis2 = cb.create_bspline_basis(rangeval=[min(location), max(location)], nbasis=Bsplines, norder=norder1)
        # Generate basic coefficient matrix
        U = inprod(fdobj1=X, fdobj2=fbasis2, Lfdobj1=0, Lfdobj2=0)
    # When X are densely measured observations
    else:
        # Transfer densely measured observations to a set of basis function
        funcX = dense_to_func(location, X, btype, nbasis1, params1, Plot=False)
        # Provide different types of basis functions
        if btype == "Bspline":
            fbasis1 = cb.create_bspline_basis(rangeval=[min(location), max(location)], nbasis=nbasis1, norder=params1)
        if btype == "Exponential":
            fbasis1 = cb.create_expon_basis(rangeval=[min(location), max(location)], nbasis=nbasis1, ratevec=params1)
        if btype == "Fourier":
            fbasis1 = cb.create_fourier_basis(rangeval=[min(location), max(location)], nbasis=nbasis1, period=params1)
        if btype == "Monomial":
            fbasis1 = cb.create_monomial_basis(rangeval=[min(location), max(location)], nbasis=nbasis1, exponents=params1)
        if btype == "Power":
            fbasis1 = cb.create_power_basis(rangeval=[min(location), max(location)], nbasis=nbasis1, exponents=params1)  
        fbasis2 = cb.create_bspline_basis(rangeval=[min(location), max(location)], nbasis=Bsplines, norder=norder1)
        n,m = X.shape
        funcCoef = funcX['coefs'].T
        basisint = inprod(fdobj1=fbasis1, fdobj2=fbasis2, Lfdobj1=0, Lfdobj2=0)
        # Generate basic coefficient matrix
        def funcU(i):
            return np.dot(funcCoef[i, :], basisint)
        U = pd.DataFrame(np.array([funcU(i) for i in range(n)]).reshape(n, -1))
    # Calculate interaction variables
    dim_G = U.shape[1]
    dim_E = z.shape[1]
    INTERACTION = np.zeros(shape=(n, dim_G * dim_E))
    k = 0
    for i in range(dim_E):
        for j in range(dim_G):
            INTERACTION[:,k] = z[:,i] * U.iloc[:,j]
            k = k + 1
    # Model for parameter initialization
    data_reg = pd.DataFrame(np.hstack((U,INTERACTION,z)))
    if ytype == 'Survival':
        model_reg = LinearRegression().fit(data_reg, y[:,0])
    if ytype == 'Binary':
        model_reg = LogisticRegression(max_iter=200).fit(data_reg, y)
    else:
        model_reg = LinearRegression().fit(data_reg, y)
    # Put basic coefficient matrix into modeling, grid search for parameters
    GridFuncGE_res = grid_scalar_ge([y,U,z], ytype, dim_G, dim_E, False, num_hidden_layers, nodes_hidden_layer,
                                    Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, t, model, 
                                    split_type, ratio, True, plot_res, model_reg, True)
    tensor1 = GridFuncGE_res[5].sparse1.weight.data.numpy()
    tensor2 = GridFuncGE_res[5].sparse2.weight.data.numpy()
    # Plot graphs of functions
    basisCoef = np.concatenate((tensor1, tensor2), axis=0).reshape(z.shape[1]+1,-1)
    betat = {f'beta{i}(t)': fd(coef = basisCoef[i,], basisobj = fbasis2) for i in range(z.shape[1]+1)}
    b = {f'b{i}': basisCoef[i,] for i in range(z.shape[1]+1)}
    if plot_beta:
        for i in range(z.shape[1]+1):
            plt.plot(location, np.array(eval_fd(location, fd(coef = basisCoef[i,], basisobj = fbasis2)))[0])
            plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
            plt.xlabel('location')
            plt.ylabel(f'beta{i}(t)')
            plt.show()
    return GridFuncGE_res, b, betat
