import numpy as np

from GENetLib.create_basis import create_bspline_basis
from GENetLib.basis_fd import basis_fd


'''Use functional data to create functional objects'''

def fd(coef=None, basisobj=None, fdnames=None):
    
    if coef is None and basisobj is None:
        basisobj = basis_fd()
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
        fdnames = dict(zip(["args", "reps", "funs"], fdnames))
    fdobj = {"coefs": coef, "basis": basisobj, "fdnames": fdnames}
    return fdobj

