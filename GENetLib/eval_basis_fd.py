import numpy as np
from getbasismatrix import getbasismatrix


def eval_basis(evalarg, basisobj, Lfdobj = 0, returnMatrix = False):
    nderiv = 0
    bwtlist = 0
    basismat = getbasismatrix(evalarg, basisobj, nderiv, returnMatrix)
    if nderiv > 0:
        nbasis = np.shape(basismat)[1]
        oneb = np.ones((1, nbasis))
        nonintwrd = False
        for j in range(nderiv):
            bfd = bwtlist[j]
            bbasis = bfd['basis']
            if bbasis['btype'] != "constant" or bfd['coefs'] != 0:
                nonintwrd = True
        if nonintwrd:
            for j in range(nderiv):
                bfd = bwtlist[j]
                if not np.all(bfd['coefs'] == 0):
                    wjarray = eval_fd(evalarg, bfd, 0, returnMatrix)
                    Dbasismat = getbasismatrix(evalarg, basisobj, j - 1, returnMatrix)
                    basismat = basismat + np.dot(wjarray, oneb) * Dbasismat
    if not returnMatrix and len(np.shape(basismat)) == 2:
        return np.asmatrix(basismat)
    else:
        return basismat

def eval_fd(evalarg, fdobj, Lfdobj = 0, returnMatrix = False):
    evaldim = np.shape(evalarg)
    if len(evaldim) >= 3:
        raise ValueError("Argument 'evalarg' is not a vector or a matrix.")
    basisobj = fdobj['basis']
    rangeval = basisobj['rangeval']
    temp = np.array(evalarg)
    temp = temp[~np.isnan(temp)]
    EPS = 5 * np.finfo(float).eps
    if np.min(temp) < rangeval[0] - EPS or np.max(temp) > rangeval[1] + EPS:
        print("Values in argument 'evalarg' are outside of permitted range and will be ignored.")
        print([rangeval[0] - np.min(temp), np.max(temp) - rangeval[1]])
    if isinstance(evalarg, list):
        n = len(evalarg)
    else:
        n = evaldim[0]
    coef = fdobj['coefs']
    coefd = np.shape(coef)
    ndim = len(coefd)
    if ndim <= 1:
        nrep = 1
    else:
        nrep = coefd[1]
    if ndim <= 2:
        nvar = 1
    else:
        nvar = coefd[2]
    if len(evaldim) > 1:
        if evaldim[1] == 1:
            evalarg = list(evalarg)
        else:
            if evaldim[1] != coefd[1]:
                raise ValueError(f"evalarg has {evaldim[1]} columns; does not match {ndim[1]} = number of columns of ffdobj$coefs")
    if ndim <= 2:
        evalarray = np.zeros((n, nrep))
    else:
        evalarray = np.zeros((n, nrep, nvar))
    if ndim == 2 or ndim == 3:
        evalarray = np.array(evalarray)
    if isinstance(evalarg, list):
        [np.nan if num < rangeval[0] - 1e-10 else num for num in evalarg]
        [np.nan if num > rangeval[1] + 1e-10 else num for num in evalarg]
        basismat = eval_basis(evalarg, basisobj, Lfdobj, returnMatrix)
        if ndim <= 2:
            evalarray = np.dot(basismat, coef)
        else:
            evalarray = np.zeros((n, nrep, nvar))
            for ivar in range(nvar):
                evalarray[:, :, ivar] = np.dot(basismat, coef[:, :, ivar])
    else:
        for i in range(nrep):
            evalargi = evalarg[:, i]
            if np.all(np.isnan(evalargi)):
                raise ValueError(f"All values are NA for replication {i}")
            index = ~np.isnan(evalargi) & (evalargi >= rangeval[0]) & (evalargi <= rangeval[1])
            evalargi = evalargi[index]
            basismat = eval_basis(evalargi, basisobj, Lfdobj, returnMatrix)
            if ndim == 2:
                evalarray[index, i] = np.dot(basismat, coef[:, i])
                evalarray[~index, i] = np.nan
            if ndim == 3:
                for ivar in range(nvar):
                    evalarray[index, i, ivar] = np.dot(basismat, coef[:, i, ivar])
                    evalarray[~index, i, ivar] = np.nan
    if len(np.shape(evalarray)) == 2 and not returnMatrix:
        return np.asmatrix(evalarray)
    else:
        return evalarray

