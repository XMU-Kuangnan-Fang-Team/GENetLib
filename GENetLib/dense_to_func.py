import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

from GENetLib.create_basis import create_bspline_basis, create_expon_basis, create_fourier_basis, create_monomial_basis, create_power_basis
from GENetLib.get_basis_matrix import get_basis_matrix
from GENetLib.fd import fd
from GENetLib.eval_basis_fd import eval_fd


'''Convert discrete observations into functional data'''

def dense_to_func(location, X, btype, nbasis, params, Plot = False):

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
        basisMatrix = get_basis_matrix(evalarg = location, basisobj = fbasis, nderiv = 0, returnMatrix = False)
        basisCoef = solve(basisMatrix.T @ basisMatrix, basisMatrix.T @ X.T)
    tofunc = fd(coef = basisCoef, basisobj = fbasis)
    if Plot:
        plt.plot(location, eval_fd(location, tofunc))
    return tofunc

