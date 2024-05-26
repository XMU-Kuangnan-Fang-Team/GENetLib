import numpy as np
import pandas as pd
from SNPgvf import SNPgvf
import CreateBasis as cb
from inprod import inprod
from GridScalerGE import GridScalerGE


def GridSNPGE(y, z, location, X, ytype, btype, num_hidden_layers, nodes_hidden_layer,
              Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, 
              nbasis1, params1, Bsplines = 20, norder1 = 4, 
              model = None, split_type = 0, ratio = [7, 3], plot = True):
    funcX = SNPgvf(location, X, btype, nbasis1, params1, Plot=False)
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
    GridSNPGE_res = GridScalerGE([y,U,z], ytype, dim_G, dim_E, False, num_hidden_layers, nodes_hidden_layer,
                                 Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, None, model, 
                                 split_type, ratio, False, plot)
    return GridSNPGE_res

