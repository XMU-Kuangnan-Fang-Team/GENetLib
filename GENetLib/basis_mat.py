import numpy as np
import pandas as pd

from GENetLib.spline_design import spline_design


'''Create difrrent types of basis matrices for functional data'''

# B-spline
def bspline_mat(x, breaks, norder=4, nderiv=0, returnMatrix=False):

    x = np.array(x)
    n = len(x)
    tol = 1e-14
    nbreaks = len(breaks)
    if nbreaks < 2:
        raise ValueError("Number of knots less than 2.")
    if min(np.diff(breaks)) < 0:
        raise ValueError("Knots are not increasing")
    if max(x) > max(breaks) + tol or min(x) < min(breaks) - tol:
        raise ValueError("Knots do not span the values of X")
    if x[n-1] > breaks[nbreaks-1]:
        breaks[nbreaks-1] = x[n-1]
    if x[0] < breaks[0]:
        breaks[0] = x[0]
    if norder > 20:
        raise ValueError("NORDER exceeds 20.")
    if norder < 1:
        raise ValueError("NORDER less than 1.")
    if nderiv > 19:
        raise ValueError("NDERIV exceeds 19.")
    if nderiv < 0:
        raise ValueError("NDERIV is negative.")
    nbasis = nbreaks + norder - 2
    if nderiv >= norder:
        return np.zeros((n, nbasis))
    knots = np.concatenate([np.repeat(breaks[0], norder - 1), breaks, np.repeat(breaks[nbreaks-1], norder - 1)])
    if nbasis >= norder:
        if nbasis > 1:
            basismat = spline_design(knots, x, norder)
        else:
            basismat = np.array(spline_design(knots, x, norder))
        if not returnMatrix and len(basismat.shape) == 2:
            return np.array(basismat)
        else:
            return basismat
    else:
        raise ValueError("NBASIS is less than NORDER.")

# Exponential function
def expon_mat(x, ratevec = [1], nderiv = 0):
    n = len(x)
    nrate = len(ratevec)
    expval = np.zeros((n, nrate))
    for irate in range(nrate):
        rate = ratevec[irate]
        expval[:, irate] = rate**nderiv * np.exp(rate * x)
    return expval

# Fourier function
def fourier_mat(x, nbasis = None , period = None , nderiv = 0):
    n = len(x)
    onen = np.ones(n)
    xrange = [np.min(x), np.max(x)]
    span = xrange[1] - xrange[0]
    if nbasis == None:
        nbasis = n
    if period == None:
        period = span
    if period <= 0:
        raise ValueError("PERIOD not positive.")
    omega = 2 * np.pi / period
    omegax = omega * x
    if nbasis <= 0:
        raise ValueError("NBASIS not positive")
    if nderiv < 0:
        raise ValueError("NDERIV is negative.")
    if nbasis % 2 == 0:
        nbasis += 1
    basismat = np.zeros((n, nbasis))
    if nderiv == 0:
        basismat[:, 0] = 1 / np.sqrt(2)
        if nbasis > 1:
            j = np.arange(2, nbasis, 2)
            k = j / 2
            args = np.outer(omegax, k)
            basismat[:, j-1] = np.sin(args)
            basismat[:, j] = np.cos(args)
    else:
        basismat[:, 0] = 0
        if nbasis > 1:
            if nderiv % 2 == 0:
                mval = nderiv / 2
                ncase = 1
            else:
                mval = (nderiv - 1) / 2
                ncase = 2
            j = np.arange(2, nbasis, 2)
            k = j / 2
            fac = np.outer(onen, ((-1)**mval) * (k * omega)**nderiv)
            args = np.outer(omegax, k)
            if ncase == 1:
                basismat[:, j-1] = fac * np.sin(args)
                basismat[:, j] = fac * np.cos(args)
            else:
                basismat[:, j-1] = fac * np.cos(args)
                basismat[:, j] = -fac * np.sin(args)
    basismat = pd.DataFrame(basismat / np.sqrt(period / 2))
    fNames = ["const"]
    n2 = nbasis // 2
    if n2 > 0:
        SC = [f"{trig}{i}" for i in range(1, n2+1) for trig in ["sin", "cos"]]
        fNames.extend(SC)
    basismat.columns = fNames
    return basismat

# Monomial function
def monomial_mat(evalarg, exponents=1, nderiv=0, argtrans=[0, 1]):
    evalarg = np.array(evalarg)
    evalarg = (evalarg - argtrans[0]) / argtrans[1]
    n = len(evalarg)
    nbasis = len(np.array(exponents))
    for ibasis in range(nbasis):
        if exponents[ibasis] - round(exponents[ibasis]) != 0:
            raise ValueError("An exponent is not an integer.")
        if exponents[ibasis] < 0:
            raise ValueError("An exponent is negative.")
    if len(exponents) > 1 and min(np.diff(np.sort(exponents))) == 0:
        raise ValueError("There are duplicate exponents.")
    monommat = np.zeros((n, nbasis))
    if nderiv == 0:
        for ibasis in range(nbasis):
            monommat[:, ibasis] = evalarg**exponents[ibasis]
    else:
        for ibasis in range(nbasis):
            print(ibasis+1)
            degree = exponents[ibasis]
            if nderiv <= degree:
                fac = degree
                if nderiv >= 2:
                    for ideriv in range(2, nderiv+1):
                        fac = fac * (degree - ideriv)
                print(fac)
                print(degree - nderiv)
                monommat[:, ibasis] = fac * evalarg**(degree - nderiv)
    return monommat

# Polynomial function
def polyg_mat(x, argvals):
    x = np.array(x)
    argvals = np.array(argvals)
    if len(argvals.shape) != 1:
        raise ValueError("ARGVALS is not a vector or 1-dim. array.")
    if np.max(x) > np.max(argvals) or np.min(x) < np.min(argvals):
        raise ValueError("ARGVALS do not span the values of X.")
    if np.min(np.diff(argvals)) <= 0:
        raise ValueError("Break-points are not strictly increasing")
    nbasis = len(argvals)
    knots = np.concatenate(([argvals[0]], argvals, [argvals[nbasis-1]]))
    basismat = spline_design(knots, x, 2)
    return basismat

# Power function
def power_mat(x, exponents, nderiv=0):
    x = np.array(x)
    n = len(x)
    nbasis = len(exponents)
    powermat = np.zeros((n, nbasis))
    if nderiv == 0:
        for ibasis in range(nbasis):
            powermat[:, ibasis] = x**exponents[ibasis]
    else:
        negative_exponent = False
        for exponent in exponents:
            if exponent - nderiv < 0:
                negative_exponent = True
                break
        if negative_exponent and any(x == 0):
            raise ValueError("A negative exponent is needed and an argument value is 0.")
        else:
            for ibasis in range(nbasis):
                degree = exponents[ibasis]
                if nderiv <= degree:
                    fac = degree
                    for ideriv in range(2, nderiv+1):
                        fac = fac * (degree - ideriv + 1)
                    powermat[:, ibasis] = fac * x**(degree - nderiv)
    return powermat

