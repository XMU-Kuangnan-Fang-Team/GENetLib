import numpy as np

from GENetLib.basis_mat import bspline_mat, expon_mat, fourier_mat, monomial_mat, polyg_mat, power_mat


'''Calculate a set of basis functions or their derivatives and a set of parameter values'''

def get_basis_matrix(evalarg, basisobj, nderiv=0, returnMatrix=False):
    if evalarg is None:
        raise ValueError("evalarg required;  is NULL.")
    evalarg = np.array(evalarg, dtype=float)
    nNA = np.sum(np.isnan(evalarg))
    if nNA > 0:
        raise ValueError(f"as.numeric(evalarg) contains {nNA} NA(s);  class(evalarg) = {type(evalarg).__name__}")
    if not isinstance(basisobj, dict):
        raise ValueError("Second argument is not a basis object.")
    if 'basisvalues' in basisobj and basisobj['basisvalues'] is not None:
        if not isinstance(basisobj['basisvalues'], (list, np.ndarray)):
            raise ValueError("BASISVALUES is not a vector.")
        basisvalues = basisobj['basisvalues']
        nvalues = len(basisvalues)
        N = len(evalarg)
        OK = False
    type_ = basisobj['btype']
    nbasis = basisobj['nbasis']
    params = basisobj['params']
    rangeval = basisobj['rangeval']
    dropind = basisobj['dropind']
    if type_ == "bspline":
        if params == []:
            breaks = [rangeval[0], rangeval[1]]
        else:
            breaks = [rangeval[0], *params, rangeval[1]]
        norder = nbasis - len(breaks) + 2
        basismat = bspline_mat(evalarg, breaks, norder, nderiv)
    elif type_ == "const":
        basismat = np.ones((len(evalarg), 1))
    elif type_ == "expon":
        basismat = expon_mat(evalarg, params, nderiv)
    elif type_ == "fourier":
        period = params[0]
        basismat = fourier_mat(evalarg, nbasis, period, nderiv)
    elif type_ == "monom":
        basismat = monomial_mat(evalarg, params, nderiv)
    elif type_ == "polygonal":
        basismat = polyg_mat(evalarg, params)
    elif type_ == "power":
        basismat = power_mat(evalarg, params, nderiv)
    else:
        raise ValueError("Basis type not recognizable")
    if len(dropind) > 0:
        basismat = np.delete(basismat, dropind, axis=1)
    if len(evalarg) == 1:
        basismat = np.asmatrix(basismat)
    if len(basismat.shape) == 2:
        return np.asmatrix(basismat)
    else:
        return np.asmatrix(basismat)

