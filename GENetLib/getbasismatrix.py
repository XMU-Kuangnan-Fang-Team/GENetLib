from BasisFunc import bspline_func, expon_func, fourier_func, monomial_func, polyg_func, power_func
import numpy as np


def getbasismatrix(evalarg, basisobj, nderiv=0, returnMatrix=False):
    if isinstance(basisobj, (int, float)) and isinstance(evalarg, dict):
        temp = basisobj
        basisobj = evalarg
        evalarg = temp
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
        for ivalues in range(nvalues):
            basisvaluesi = basisvalues[ivalues]
            if not isinstance(basisvaluesi, (list, np.ndarray)):
                raise ValueError("BASISVALUES does not contain lists.")
            argvals = basisvaluesi[0]
            if len(basisvaluesi) >= nderiv + 2:
                if N == len(argvals):
                    if np.all(argvals == evalarg):
                        basismat = basisvaluesi[nderiv + 1]
                        OK = True
        if OK:
            if len(basismat.shape) == 2:
                return np.asmatrix(basismat)
            return basismat
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
        basismat = bspline_func(evalarg, breaks, norder, nderiv)
    elif type_ == "const":
        basismat = np.ones((len(evalarg), 1))
    elif type_ == "expon":
        basismat = expon_func(evalarg, params, nderiv)
    elif type_ == "fourier":
        period = params[0]
        basismat = fourier_func(evalarg, nbasis, period, nderiv)
    elif type_ == "monom":
        basismat = monomial_func(evalarg, params, nderiv)
    elif type_ == "polygonal":
        basismat = polyg_func(evalarg, params)
    elif type_ == "power":
        basismat = power_func(evalarg, params, nderiv)
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

