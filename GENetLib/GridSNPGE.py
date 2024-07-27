import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from GENetLib.SNPgvf import SNPgvf
import GENetLib.CreateBasis as cb
from GENetLib.inprod import inprod
from GENetLib.GridScalerGE import GridScalerGE
from GENetLib.eval_basis_fd import eval_fd
from GENetLib.FD import FD


def GridSNPGE(y, z, location, X, ytype, btype, num_hidden_layers, nodes_hidden_layer,
              Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs,
              nbasis1, params1, t = None, Bsplines = 20, norder1 = 4, 
              model = None, split_type = 0, ratio = [7, 3], plot_res = True, plot_beta = True):
    
    funcX = SNPgvf(location, X, btype, nbasis1, params1, Plot = False)
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
    data_reg = pd.DataFrame(np.hstack((U,INTERACTION,z)))
    if ytype == 'Survival':
        model_reg = LinearRegression().fit(data_reg, y[:,0])
    else:
        model_reg = LinearRegression().fit(data_reg, y)
    GridSNPGE_res = GridScalerGE([y,U,z], ytype, dim_G, dim_E, False, num_hidden_layers, nodes_hidden_layer,
                                 Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, t, model, 
                                 split_type, ratio, True, plot_res, model_reg, True)
    tensor1 = GridSNPGE_res[5].sparse1.weight.data.numpy()
    tensor2 = GridSNPGE_res[5].sparse2.weight.data.numpy()
    basisCoef = np.concatenate((tensor1, tensor2), axis=0).reshape(z.shape[1]+1,-1)
    betat = {f'beta{i}(t)': FD(coef = basisCoef[i,], basisobj = fbasis2) for i in range(z.shape[1]+1)}
    b = {f'b{i}': basisCoef[i,] for i in range(z.shape[1]+1)}
    if plot_beta:
        for i in range(z.shape[1]+1):
            plt.plot(location, np.array(eval_fd(location, FD(coef = basisCoef[i,], basisobj = fbasis2)))[0])
            plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
            plt.xlabel('location')
            plt.ylabel(f'beta{i}(t)')
            plt.show()
    return GridSNPGE_res, b, betat
