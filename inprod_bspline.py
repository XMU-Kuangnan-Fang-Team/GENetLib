import numpy as np
import pandas as pd

from GENetLib.PPFunc import ppBspline, ppDeriv
from GENetLib.polyprod import polyprod


def inprod_bspline(fdobj1, fdobj2=None, nderiv1=0, nderiv2=0):
    
    def outer_product(a, b):
        a = np.array(a)[:, np.newaxis]
        b = np.array(b)[np.newaxis, :]
        outer = a * b
        return outer
    if fdobj2 is None:
        fdobj2 = fdobj1
    basis1 = fdobj1['basis']
    type1 = basis1['btype']
    if type1 != "bspline":
        raise Exception("FDOBJ1 does not have a B-spline basis.")
    range1 = basis1['rangeval']
    breaks1 = np.concatenate(([range1[0]], basis1['params'], [range1[1]]))
    nbasis1 = basis1['nbasis']
    norder1 = nbasis1 - len(breaks1) + 2
    basis2 = fdobj2['basis']
    type2 = basis2['btype']
    if type2 != "bspline":
        raise Exception("FDOBJ2 does not have a B-spline basis.")
    range2 = basis2['rangeval']
    breaks2 = np.concatenate(([range2[0]], basis2['params'], [range2[1]]))
    nbasis2 = basis2['nbasis']
    norder2 = nbasis2 - len(breaks2) + 2
    if any(x != y for x, y in zip(range1, range2)):
        raise Exception("The argument ranges for FDOBJ1 and FDOBJ2 are not identical.")
    if len(breaks1) != len(breaks2):
        raise Exception("The numbers of knots for FDOBJ1 and FDOBJ2 are not identical")
    if any(x != y for x, y in zip(breaks1, breaks2)):
        raise Exception("The knots for FDOBJ1 and FDOBJ2 are not identical.")
    else:
        breaks = breaks1
    if len(breaks) < 2:
        raise Exception("The length of argument BREAKS is less than 2.")
    breakdiff = np.diff(breaks)
    if min(breakdiff) <= 0:
        raise Exception("Argument BREAKS is not strictly increasing.")
    coef1 = pd.DataFrame(fdobj1['coefs']).T
    coef2 = pd.DataFrame(fdobj2['coefs']).T
    if len(coef1.shape) > 2:
        raise Exception("FDOBJ1 is not univariate.")
    if len(coef2.shape) > 2:
        raise Exception("FDOBJ2 is not univariate.")
    nbreaks = len(breaks)
    ninterval = nbreaks - 1
    nbasis1 = ninterval + norder1 - 1
    nbasis2 = ninterval + norder2 - 1
    if coef1.shape[1] != nbasis1 or coef2.shape[1] != nbasis2:
        raise Exception("Error: coef1 should have length no. breaks1+norder1-2 and coef2 no. breaks2+norder2-2.")
    breaks1 = breaks[0]
    breaksn = breaks[nbreaks-1]
    temp = breaks[1:(nbreaks - 1)]
    knots1 = np.concatenate((breaks1 * np.ones(norder1), temp, breaksn * np.ones(norder1)))
    knots2 = np.concatenate((breaks1 * np.ones(norder2), temp, breaksn * np.ones(norder2)))
    nrep1 = coef1.shape[0]
    polycoef1 = np.zeros((ninterval, norder1 - nderiv1, nrep1))
    for i in range(nbasis1):
        ppBlist = ppBspline(knots1[i:(i + norder1 + 1)])
        Coeff = ppBlist[0]
        index = ppBlist[1]
        index = index + i - norder1
        CoeffD = ppDeriv(Coeff, nderiv1)
        if nrep1 == 1:
            polycoef1[index, :, 0] = outer_product(coef1.iloc[:,i], CoeffD) + polycoef1[index, :, 0]
        else:
            for j in range(len(index)):
                temp = outer_product(CoeffD.iloc[j, :], coef1.iloc[:,i])
                polycoef1[index[j], :, :] = temp + polycoef1[index[j], :, :]
    nrep2 = coef2.shape[0]
    polycoef2 = np.zeros((ninterval, norder2 - nderiv2, nrep2))
    for i in range(nbasis2):
        ppBlist = ppBspline(knots2[i:(i + norder2 + 1)])
        Coeff = ppBlist[0]
        index = ppBlist[1]
        index = index + i - norder2
        CoeffD = ppDeriv(Coeff, nderiv2)
        if nrep2 == 1:
            polycoef2[index, :, 0] = outer_product(coef2.iloc[:,i], CoeffD) + polycoef2[index, :, 0]
        else:
            for j in range(len(index)):
                temp = outer_product(CoeffD.iloc[j, :], coef2.iloc[:,i])
                polycoef2[index[j], :, :] = temp + polycoef2[index[j], :, :]
    prodmat = np.zeros((nrep1, nrep2))
    for j in range(ninterval):
        c1 = np.asmatrix(polycoef1[j, :, :])
        c2 = np.asmatrix(polycoef2[j, :, :])
        polyprodmat = polyprod(c1, c2)
        N = polyprodmat.shape[2]
        delta = breaks[j + 1] - breaks[j]
        power = delta
        prodmati = np.zeros((nrep1, nrep2))
        for i in range(1,N+1):
            prodmati = prodmati + power * polyprodmat[:, :, N - i] / i
            power = power * delta
        prodmat = prodmat + prodmati
    return prodmat
    
