import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from SNPgvf import SNPgvf
import CreateBasis as cb
from inprod import inprod
from GridScalerGE import GridScalerGE
from eval_basis_fd import eval_fd
from FD import FD


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


'''test'''
'''
from SimDataSNP import SimDataSNP
num_hidden_layers = 2
nodes_hidden_layer = [100, 10]
Learning_Rate2 = [0.005, 0.01, 0.015]
L2 = [0.005, 0.01, 0.015]
Learning_Rate1 = [0.001, 0.005]
L = [0.005, 0.006, 0.007]
Num_Epochs = 50
nbasis1 = 5
params1 = 4
snp_continuous = SimDataSNP(1000, 30, 'Continuous', seed = 1)
y = snp_continuous['y']
z = snp_continuous['z']
location = snp_continuous['location']
X = snp_continuous['X']
GridSNPGE_Res2 = GridSNPGE(y, z, location, X, 'Continuous', 'Bspline', num_hidden_layers, nodes_hidden_layer,
                           Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, 
                           nbasis1, params1, Bsplines = 5, norder1 = 4, 
                           model = None, split_type = 0, ratio = [7,3], plot_res = True)
print(GridSNPGE_Res2[0][5].sparse1.weight.data)
print(GridSNPGE_Res2[0][5].sparse2.weight.data)
'''
'''
from SimDataSNP import SimDataSNP
num_hidden_layers = 2
nodes_hidden_layer = [50,10]
Learning_Rate2 = [0.001, 0.005, 0.01]
L2 = [0.001, 0.005, 0.01]
Learning_Rate1 = [0.001, 0.005, 0.01]
L = [0.005, 0.006, 0.007]
Num_Epochs = 100
nbasis1 = 5
params1 = 4
seed = 123
snp_survival = SimDataSNP(1000, 30, 'Survival', seed = seed)
y = snp_survival['y']
z = snp_survival['z']
location = list(snp_survival['location'])
X = snp_survival['X']
GridSNPGE_Res1 = GridSNPGE(y, z, location, X, 'Survival', 'Bspline', num_hidden_layers, nodes_hidden_layer,
                           Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, 
                           nbasis1, params1, Bsplines = 5, norder1 = 4, 
                           model = None, split_type = 0, ratio = [7, 3], plot_res = True)
'''
'''
snp_binary = SimDataSNP(1500, 50, 'Binary')
y = list(snp_binary['y'])
z = snp_binary['z']
location = list(snp_binary['location'])
X = snp_binary['X']
params1 =[-1,-0.5,0,0.5,1]
GridSNPGE_Res3 = GridSNPGE(y, z, location, X, 'Binary', 'Power', num_hidden_layers, nodes_hidden_layer,
                           Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, 
                           nbasis1, params1, Bsplines = 5, norder1 = 4, 
                           model = None, split_type = 0, ratio = [7, 3], plot_res = True)
'''
