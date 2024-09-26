import numpy as np

from GENetLib.knotmultchk import knotmultchk
from GENetLib.fd_chk import fd_chk
from GENetLib.create_basis import create_bspline_basis, create_fourier_basis, create_constant_basis
from GENetLib.fd import fd
from GENetLib.eval_basis_fd import eval_fd


'''Calculate the inner product of function data objects'''

def inprod(fdobj1, fdobj2 = None, Lfdobj1 = 0, Lfdobj2 = 0, rng = None, wtfd = 0):

    result1 = fd_chk(fdobj1)
    nrep1 = result1[0]
    fdobj1 = result1[1]
    coef1 = fdobj1['coefs']
    basisobj1 = fdobj1['basis']
    btype1 = basisobj1['btype']
    range1 = basisobj1['rangeval']
    if rng == None:
        rng = range1
    if fdobj2 is None:
        tempfd = fdobj1
        tempbasis = tempfd['basis']
        temptype = tempbasis['btype']
        temprng = tempbasis['rangeval']
        if temptype == "bspline":
            basis2 = create_bspline_basis(temprng, 1, 1)
        else:
            if temptype == "fourier":
                basis2 = create_fourier_basis(temprng, 1)
            else:
                basis2 = create_constant_basis(temprng)
        fdobj2 = fd(np.array([1]).reshape(-1,1), basis2)
    result2 = fd_chk(fdobj2)
    nrep2 = result2[0]
    fdobj2 = result2[1]
    coef2 = fdobj2['coefs']
    basisobj2 = fdobj2['basis']
    btype2 = basisobj2['btype']
    if rng[0] < range1[0] or rng[1] > range1[1]:
        raise ValueError("Limits of integration are inadmissible.")
    iter = 0
    rngvec = rng
    knotmult = []
    if btype1 == "bspline":
        knotmult = knotmultchk(basisobj1, knotmult)
    if btype2 == "bspline":
        knotmult = knotmultchk(basisobj2, knotmult)
    if len(knotmult) > 0:
        knotmult = sorted(set(knotmult))
        knotmult = [k for k in knotmult if k > rng[0] and k < rng[1]]
        rngvec = [rng[0]] + knotmult + [rng[1]]
    if np.all(coef1 == 0) or np.all(coef2 == 0):
        return np.zeros((nrep1, nrep2))
    JMAX = 15
    JMIN = 5
    EPS = 1e-04
    inprodmat = np.zeros((nrep1, nrep2))
    nrng = len(rngvec)
    for irng in range(1, nrng):
        rngi = [rngvec[irng - 1], rngvec[irng]]
        if irng > 2:
            rngi[0] += 1e-10
        if irng < nrng:
            rngi[1] -= 1e-10
        iter = 1
        width = rngi[1] - rngi[0]
        JMAXP = JMAX + 1
        h = [1] * JMAXP
        h[1] = 0.25
        s = np.zeros((JMAXP, nrep1, nrep2))
        fx1 = eval_fd(rngi, fdobj1, Lfdobj1)
        fx2 = eval_fd(rngi, fdobj2, Lfdobj2)
        if not isinstance(wtfd, (int, float)):
            wtd = eval_fd(rngi, wtfd, 0)
            fx2 = np.multiply(np.reshape(wtd, (len(wtd), len(fx2[0]))), fx2)
        s[0, :, :] = width * np.dot(fx1.T, fx2) / 2
        tnm = 0.5
        for iter in range(1, JMAX):
            tnm *= 2
            if iter == 1:
                x = [np.mean(rngi)]
            else:
                del_ = width / tnm
                x = list(np.arange(rngi[0] + del_ / 2, rngi[1] - del_ / 2, del_))
            fx1 = eval_fd(x, fdobj1, Lfdobj1)
            fx2 = eval_fd(x, fdobj2, Lfdobj2)
            if not isinstance(wtfd, (int, float)):
                wtd = eval_fd(wtfd, x, 0)
                fx2 = np.multiply(np.reshape(wtd, (len(wtd), len(fx2[0]))), fx2)
            s[iter, :, :] = (s[iter - 1, :, :] + width * np.dot(fx1.T, fx2) / tnm) / 2
            if iter >= 4:
                ind = list(range(iter - 4, iter + 1))
                ya = s[ind, :, :]
                xa = h[iter - 4:iter + 1]
                absxa = np.abs(xa)
                ns = np.argmin(absxa)
                cs = ya.copy()
                ds = ya.copy()
                y = ya[ns, :, :]
                ns -= 1
                for m in range(1, 5):
                    for i in range(5 - m):
                        ho = xa[i]
                        hp = xa[i + m]
                        w = (cs[i + 1, :, :] - ds[i, :, :]) / (ho - hp)
                        ds[i, :, :] = hp * w
                        cs[i, :, :] = ho * w
                    if 2 * ns < 5 - m:
                        dy = cs[ns, :, :]
                    else:
                        dy = ds[ns - 1, :, :]
                        ns -= 1
                    y += dy
                ss = y
                errval = np.max(np.abs(dy))
                ssqval = np.max(np.abs(ss))
                if np.all(ssqval > 0):
                    crit = errval / ssqval
                else:
                    crit = errval
                if crit < EPS and iter >= JMIN:
                    break
            s[iter + 1, :, :] = s[iter, :, :]
            h[iter + 1] = 0.25 * h[iter]
            if iter == JMAX:
                print("Warning: Failure to converge.")
        inprodmat += ss
    if len(inprodmat.shape) == 2:
        return np.asmatrix(inprodmat)
    else:
        return inprodmat
    
