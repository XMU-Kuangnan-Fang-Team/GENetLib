from BasisFD import BasisFD
import numpy as np
from CreateBasis import create_bspline_basis


def FD(coef=None, basisobj=None, fdnames=None):
    if coef is None and basisobj is None:
        basisobj = BasisFD()
    if coef is None:
        coef = [0]*basisobj['nbasis']
    btype = basisobj['btype']
    if isinstance(coef, list):
        coef = np.array(coef)
        if btype == "constant":
            coef = coef.T
        coefd = coef.reshape(len(coef),-1).shape
        ndim = len(coefd)
    elif isinstance(coef, np.ndarray):
        coefd = coef.reshape(len(coef),-1).shape
        ndim = len(coefd)
    else:
        raise ValueError("Type of 'coef' is not correct")
    if ndim > 3:
        raise ValueError("'coef' not of dimension 1, 2 or 3")
    if basisobj is None:
        rc = [min(coef), max(coef)]
        if rc[1] - rc[0] == 0:
            rc = [rc[0], rc[0]+1]
        dimC = coef.shape
        nb = len(coef) if dimC is None else dimC[0]
        basisobj = create_bspline_basis(rc, nbasis=max(4, nb))
        btype = basisobj.btype
    nbasis = basisobj['nbasis']
    ndropind = len(basisobj['dropind'])
    if coefd[0] != nbasis - ndropind:
        raise ValueError("First dim. of 'coef' not equal to 'nbasis - ndropind'.")
    nrep = coefd[1] if ndim > 1 else 1
    nvar = coefd[2] if ndim > 2 else 1
    if fdnames is None:
        if ndim == 1:
            fdnames = ["time", "reps", "values"]
        if ndim == 2:
            fdnames1 = ["reps"+str(i+1) for i in range(nrep)]
            fdnames = ["time"] + [fdnames1] + ["values"]
        if ndim == 3:
            fdnames1 = ["reps"+str(i+1) for i in range(nrep)]
            fdnames2 = ["values"+str(i+1) for i in range(nvar)]
            fdnames = ["time"] + [fdnames1] + [fdnames2]
        fdnames = dict(zip(["args", "reps", "funs"], fdnames))
    if coef.ndim is None:
        dimc = coef.shape
        ndim = len(dimc)
        dnms = [None]*ndim
        if dimc[0] == len(fdnames["args"]):
            dnms[0] = fdnames["args"]
        if ndim > 1 and dimc[1] == len(fdnames["reps"]):
            dnms[1] = fdnames["reps"]
        if ndim > 2 and dimc[2] == len(fdnames["funs"]):
            dnms[2] = fdnames["funs"]
        if any(d is not None for d in dnms):
            coef = np.array(coef, dtype={'names':dnms, 'formats':['f8']*len(dnms)})
    fdobj = {"coefs": coef, "basis": basisobj, "fdnames": fdnames}
    return fdobj