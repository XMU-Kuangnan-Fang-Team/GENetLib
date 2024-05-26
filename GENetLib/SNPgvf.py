import numpy as np
from scipy.linalg import solve
from CreateBasis import create_bspline_basis, create_expon_basis, create_fourier_basis, create_monomial_basis, create_power_basis
from getbasismatrix import getbasismatrix
from FD import FD
import matplotlib.pyplot as plt
from eval_basis_fd import eval_fd


def SNPgvf(location, X, btype, nbasis, params, Plot = False):
    if not isinstance(location, list):
        raise ValueError("location should be of list type.")
    if btype == "Bspline":
        fbasis = create_bspline_basis(rangeval = [np.min(location), np.max(location)], nbasis = nbasis, norder = params)
    if btype == "Exponential":
        fbasis = create_expon_basis(rangeval = [np.min(location), np.max(location)], nbasis = nbasis, ratevec = params)
    if btype == "Fourier":
        fbasis = create_fourier_basis(rangeval = [np.min(location), np.max(location)], nbasis = nbasis, period = params)
    if btype == "Monomial":
        fbasis = create_monomial_basis(rangeval = [np.min(location), np.max(location)], nbasis = nbasis, exponents = params)
    if btype == "Power":
        fbasis = create_power_basis(rangeval = [np.min(location), np.max(location)], nbasis = nbasis, exponents = params)
    n, m = X.shape
    truelengths = np.count_nonzero(~np.isnan(X))
    if truelengths == n * m:
        basisMatrix = getbasismatrix(evalarg = location, basisobj = fbasis, nderiv = 0, returnMatrix = False)
        basisCoef = solve(basisMatrix.T @ basisMatrix, basisMatrix.T @ X.T)
    if truelengths != n * m:
        location_list = [location[~np.isnan(X[i, :])] for i in range(n)]
        X_list = [X[i, ~np.isnan(X[i, :])] for i in range(n)]

        def coefFunc(i):
            basisMatrix = getbasismatrix(evalarg = location_list[i], basisobj = fbasis, nderiv = 0, returnMatrix = False)
            basiscoef = solve(basisMatrix.T @ basisMatrix, basisMatrix.T @ X_list[i])
            return basiscoef

        basisCoef = np.array([coefFunc(i) for i in range(n)])
    gvf = FD(coef = basisCoef, basisobj = fbasis)
    if Plot:
        plt.plot(location, eval_fd(location, gvf))
    return gvf

