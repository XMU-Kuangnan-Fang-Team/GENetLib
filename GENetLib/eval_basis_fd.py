import numpy as np

from GENetLib.get_basis_matrix import get_basis_matrix
from GENetLib.create_basis import create_constant_basis
from GENetLib.fd import fd


'''Calculate the value of basis functions and functional objects'''

def lfd(nderiv = 0, bwtlist = None):
    if not isinstance(nderiv, int):
        raise ValueError("Order of operator is not numeric.")
    if nderiv < 0:
        raise ValueError("Order of operator is negative.")
    if bwtlist is None:
        bwtlist = []
    if bwtlist == None:
        bwtlist = [None] * nderiv
        if nderiv > 0:
            conbasis = create_constant_basis()
            bwtlist = [fd(0, conbasis) for _ in range(nderiv)]
    nbwt = len(bwtlist)
    if nbwt != nderiv and nbwt != nderiv + 1:
        raise ValueError("The size of bwtlist is inconsistent with nderiv.")
    if nderiv > 0:
        rangevec = [0, 1]
        for j in range(nbwt):
            bfdj = bwtlist[j]
            bbasis = bfdj['basis']
            rangevec = bbasis['rangeval']
            btype = bbasis['btype']
            if btype != "const":
                brange = bbasis['rangeval']
                if rangevec != brange:
                    raise ValueError("Ranges are not compatible.")
    Lfdobj = {'nderiv': nderiv, 'bwtlist': bwtlist}
    return Lfdobj

def int2lfd(m = 0):
    if m < 0:
        raise ValueError("Argument is negative.")
    if m == 0:
        bwtlist = None
    else:
        basisobj = create_constant_basis([0, 1])
        bwtlist = [fd([0], basisobj) for _ in range(m)]
    Lfdobj = lfd(m, bwtlist)
    return Lfdobj

# Basis functions
def eval_basis(evalarg, basisobj, Lfdobj = 0, returnMatrix = False):
    if isinstance(Lfdobj, int):
        Lfdobj = int2lfd(Lfdobj)
    nderiv = Lfdobj['nderiv']
    bwtlist = Lfdobj['bwtlist']
    basismat = get_basis_matrix(evalarg, basisobj, nderiv, returnMatrix)
    if nderiv > 0:
        nbasis = basismat.shape[1]
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
                    Dbasismat = get_basis_matrix(evalarg, basisobj, j, returnMatrix)
                    basismat = basismat + (np.array(wjarray).T @ oneb) * np.array(Dbasismat)
    if not returnMatrix and len(basismat.shape) == 2:
        return np.asmatrix(basismat)
    else:
        return basismat

# Functional objects
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
    if len(np.shape(evalarray)) == 2 and not returnMatrix:
        return np.asmatrix(evalarray)
    else:
        return evalarray

