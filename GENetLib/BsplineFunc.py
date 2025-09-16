import numpy as np
from GENetLib.fda_func import fd, inprod, eval_basis, int2lfd
from GENetLib.fda_func import ppbspline, ycheck, wtcheck, fdparcheck
from scipy.linalg import cholesky, solve, qr, eigh


# Create a class for b-spline functions
class BsplineFunc:
    def __init__(self, basisobj, Lfdobj=2, rng=None, returnMatrix=False):
        self.basisobj = basisobj
        self.Lfdobj = Lfdobj
        self.rng = rng
        self.returnMatrix = returnMatrix

    # Calculate B-spline penaty matrix
    def penalty_matrix(self, btype="spline"):
        if btype == "spline":

            # Initialize parameters
            if type(self.Lfdobj) == int:
                Lfdobj = int2lfd(self.Lfdobj)
            else:
                Lfdobj = self.Lfdobj
            nbasis = self.basisobj["nbasis"]
            params = self.basisobj["params"]
            Rangeval = self.basisobj["rangeval"]
            Rng = self.rng if self.rng is not None else Rangeval
            rng = np.array(Rng, dtype=float)
            rangeval = np.array(Rangeval, dtype=float)
            params = np.array(params, dtype=float)
            breaks = np.concatenate(([rangeval[0]], params, [rangeval[1]]))
            nbreaks = len(breaks)
            ninterval = nbreaks - 1
            nderiv = Lfdobj["nderiv"]
            norder = nbasis - len(params)
            if nderiv >= norder:
                raise ValueError(
                    f"Order {nderiv} is greater than or equal to B-splines of order {norder}"
                )
            if nderiv > 0 and nderiv == norder - 1:
                raise ValueError(
                    f"Penalty matrix cannot be evaluated for derivative of order {nderiv} for B-splines of order {norder}"
                )
            bwtlist = Lfdobj["bwtlist"]
            isintLfd = True
            if nderiv > 0:
                for ideriv in range(1, nderiv + 1):
                    fdj = bwtlist[ideriv - 1]
                    if fdj is not None and not np.all(fdj["coefs"] == 0):
                        isintLfd = False
                        break
            intbreaks = np.concatenate(([rng[0]], params, [rng[1]]))
            index = (intbreaks >= rng[0]) & (intbreaks <= rng[1])
            intbreaks = intbreaks[index]
            uniquebreaks = np.min(np.diff(intbreaks)) > 0
            if (
                isintLfd
                and rng[0] == rangeval[0]
                and uniquebreaks
                and rng[1] == rangeval[1]
            ):
                onesv = np.ones((1, norder))
                knots = np.concatenate(
                    (
                        (rangeval[0] * onesv)[0],
                        breaks[1 : nbreaks - 1],
                        (rangeval[1] * onesv)[0],
                    )
                )
                polyorder = norder - nderiv
                ndegree = polyorder - 1
                prodorder = 2 * ndegree + 1
                polycoef = np.zeros((ninterval, polyorder, norder))
                indxdown = list(range(norder, nderiv, -1))
                for i in range(nbasis):
                    t = knots[i : i + norder + 1]
                    ppBlist = ppbspline(t)
                    Coeff = ppBlist[0]
                    index = ppBlist[1]
                    nrowcoef = Coeff.shape[0]
                    index = index + i - norder
                    CoeffD = np.array(Coeff)[:, :polyorder]
                    if nderiv > 0:
                        for ideriv in range(1, nderiv + 1):
                            fac = np.array(indxdown) - ideriv
                            CoeffD = np.outer(np.ones(nrowcoef), fac) * CoeffD
                    if i >= norder - 1:
                        k = norder - 1
                    else:
                        k = i
                    if i <= norder - 1:
                        m = i
                    else:
                        m = norder - 1
                    for j in range(nrowcoef):
                        polycoef[i - k + j, :, m - j] = CoeffD[j, :]
                prodmat = np.zeros((nbasis, nbasis))
                convmat = np.zeros((norder, norder, prodorder))
                for k in range(ninterval):
                    Coeff = polycoef[k, :, :]
                    for i in range(ndegree):
                        ind = np.arange(i + 1)
                        if len(ind) == 1:
                            convmat[:, :, i] = np.outer(
                                Coeff[ind, :], Coeff[i - ind, :]
                            )
                            convmat[:, :, prodorder - i - 1] = np.outer(
                                Coeff[ndegree - ind, :],
                                Coeff[ndegree - i + ind, :],
                            )
                        else:
                            convmat[:, :, i] = np.dot(
                                Coeff[ind, :].T, Coeff[i - ind, :]
                            )
                            convmat[:, :, prodorder - i - 1] = np.dot(
                                Coeff[ndegree - ind - 1, :].T,
                                Coeff[ndegree - i + ind, :],
                            )
                    ind = np.arange(ndegree + 1)
                    convmat[:, :, ndegree] = np.dot(
                        Coeff[ind, :].T, Coeff[ndegree - ind, :]
                    )
                    delta = breaks[k + 1] - breaks[k]
                    power = delta
                    prodmati = np.zeros((norder, norder))
                    for i in range(1, prodorder + 1):
                        prodmati += power * convmat[:, :, prodorder - i] / i
                        power *= delta
                    index = np.arange(k, k + norder)
                    prodmat[index[:, None], index] += prodmati
                penaltymat = prodmat
            else:
                if uniquebreaks:
                    prodmat = inprod(
                        self.basisobj, self.basisobj, Lfdobj, Lfdobj, rng
                    )
                else:
                    rngvec = [rng[0]]
                    for i in range(1, nbreaks):
                        if breaks[i] == breaks[i - 1]:
                            rngvec.append(breaks[i])
                    rngvec = list(set(rngvec))
                    nrng = len(rngvec)
                    if rngvec[-1] < rng[1]:
                        rngvec.append(rng[1])
                        nrng += 1
                    prodmat = np.zeros((nbasis, nbasis))
                    for i in range(1, nrng):
                        rngi = [rngvec[i - 1] + 1e-10, rngvec[i] - 1e-10]
                        prodmati = inprod(
                            self.basisobj, self.basisobj, Lfdobj, Lfdobj, rngi
                        )
                        prodmat += prodmati
                penaltymat = prodmat
            return penaltymat

        elif btype == "fourier":
            nbasis = self.basisobj["nbasis"]
            if nbasis % 2 == 0:
                self.basisobj["nbasis"] = nbasis + 1
            type_ = self.basisobj["btype"]
            if type_ != "fourier":
                raise ValueError("Wrong basis type")
            if isinstance(self.Lfdobj, int):
                Lfdobj_ = int2lfd(self.Lfdobj)
                nderiv = Lfdobj_["nderiv"]
                penaltymatrix = inprod(
                    self.basisobj, self.basisobj, Lfdobj_, Lfdobj_
                )
            else:
                nderiv = self.Lfdobj["nderiv"]
                penaltymatrix = inprod(
                    self.basisobj, self.basisobj, self.Lfdobj, self.Lfdobj
                )
            return penaltymatrix

    def smooth_basis(
        self,
        argvals,
        y,
        wtvec=None,
        fdnames=None,
        covariates=None,
        dfscale=1,
        returnMatrix=False,
    ):

        # Check and initialize parameters
        fdParobj = self.basisobj
        dimy = y.shape
        n = dimy[0]
        y_check = ycheck(y, n)
        y = y_check["y"]
        y0 = y
        nrep = y_check["ncurve"]
        nvar = y_check["nvar"]
        fdParobj = fdparcheck(fdParobj, nrep)
        fdobj = fdParobj["fd"]
        lambda_ = fdParobj["lambda"]
        Lfdobj = fdParobj["lfd"]
        penmat = fdParobj["penmat"]
        if lambda_ < 0:
            print("Value of 'lambda' was negative. 0 used instead.")
            lambda_ = 0
        wtlist = wtcheck(n, wtvec)
        wtvec = wtlist["wtvec"]
        onewt = wtlist["onewt"]
        matwt = wtlist["matwt"]
        nderiv = Lfdobj["nderiv"]
        basisobj = fdobj["basis"]
        dropind = basisobj["dropind"]
        ndropind = len(dropind) if dropind is not None else 0
        nbasis = basisobj["nbasis"] - ndropind
        names = basisobj["names"]
        if ndropind > 0:
            names = np.delete(names, dropind)
        if len(dimy) == 2:
            coef = np.zeros((nbasis, dimy[1]))
            ynames = dimy[1]
            vnames = "value"
        elif len(dimy) == 3:
            coef = np.zeros((nbasis, dimy[1], dimy[2]))
            ynames = dimy[1]
            vnames = dimy[2]
        if covariates is not None:
            if not np.issubdtype(covariates.dtype, np.number):
                raise ValueError("Optional argument COVARIATES is not numeric.")
            if covariates.shape[0] != n:
                raise ValueError(
                    "Optional argument COVARIATES has incorrect number of rows."
                )
            q = covariates.shape[1]
        else:
            q = 0
            beta_ = None
        tnames = None
        if y.ndim > 0 and y.shape[0] > 0:
            tnames = np.arange(1, n + 1)
        basismat = eval_basis(argvals, basisobj, 0, returnMatrix)

        # Calculation
        if n > nbasis + q or lambda_ > 0:
            if covariates is not None:
                ind1 = np.arange(n)
                ind2 = np.arange(nbasis, nbasis + q)
                basismat = np.asmatrix(basismat)
                basismat = np.c_[basismat, np.zeros((basismat.shape[0], q))]
                basismat[ind1[:, np.newaxis], ind2] = covariates
            if matwt:
                wtfac = cholesky(wtvec)
                basisw = wtvec @ basismat
            else:
                rtwtvec = np.sqrt(wtvec)
                basisw = (wtvec @ np.ones((1, nbasis + q))) * np.array(basismat)
            Bmat = basisw.T @ basismat
            Bmat0 = Bmat
            if len(dimy) < 3:
                Dmat = basisw.T @ y
            else:
                Dmat = np.zeros((nbasis + q, nrep, nvar))
                for ivar in range(nvar):
                    Dmat[:, :, ivar] = basisw.T @ y[:, :, ivar]
            if lambda_ > 0:
                if penmat is None:
                    penmat = BsplineFunc(
                        basisobj=basisobj, Lfdobj=Lfdobj
                    ).penalty_matrix()
                Bnorm = np.sqrt(np.sum(np.diag(Bmat0.T @ Bmat0)))
                pennorm = np.sqrt(np.sum(penmat * penmat))
                condno = pennorm / Bnorm
                if lambda_ * condno > 1e12:
                    lambda_ = 1e12 / condno
                    print(
                        f"Warning: lambda reduced to {lambda_} to prevent overflow"
                    )
                if covariates is not None:
                    penmat = np.block(
                        [
                            [penmat, np.zeros((nbasis, q))],
                            [np.zeros((q, nbasis)), np.zeros((q, q))],
                        ]
                    )
                Bmat = Bmat0 + lambda_ * penmat
            else:
                penmat = None
                Bmat = Bmat0
            Bmat = (Bmat + Bmat.T) / 2
            Lmat = cholesky(Bmat)
            Lmatinv = solve(Lmat, np.eye(Lmat.shape[0]))
            Bmatinv = Lmatinv @ Lmatinv.T

            if len(dimy) < 3:
                coef = Bmatinv @ Dmat
                if covariates is not None:
                    beta_ = coef[nbasis:, :]
                    coef = coef[:nbasis, :]
                else:
                    beta_ = None
            else:
                coef = np.zeros((nbasis, nrep, nvar))
                if covariates is not None:
                    beta_ = np.zeros((q, nrep, nvar))
                else:
                    beta_ = None
                for ivar in range(nvar):
                    coefi = Bmatinv @ Dmat[:, :, ivar]
                    if covariates is not None:
                        beta_[:, :, ivar] = coefi[nbasis:, :]
                        coef[:, :, ivar] = coefi[:nbasis, :]
                    else:
                        coef[:, :, ivar] = coefi
        else:
            if n == nbasis + q:
                if len(dimy) == 2:
                    coef = np.linalg.solve(basismat, y)
                else:
                    for ivar in range(nvar):
                        coef[:, :, ivar] = np.linalg.solve(
                            basismat, y[:, :, ivar]
                        )
                penmat = None
            else:
                raise ValueError(
                    f"The number of basis functions = {nbasis + q} exceeds {n} = the number of points to be smoothed."
                )
        if onewt:
            temp = basismat.T @ basismat
            if lambda_ > 0:
                temp = temp + lambda_ * penmat
            L = cholesky(temp)
            MapFac = solve(L.T, basismat.T)
            y2cMap = solve(L, MapFac)
        else:
            if matwt:
                temp = basismat.T @ (wtvec @ basismat)
            else:
                temp = basismat.T @ (wtvec * basismat)

            if lambda_ > 0:
                temp = temp + lambda_ * penmat
            L = cholesky((temp + temp.T) / 2)
            MapFac = solve(L.T, basismat.T)
            if matwt:
                y2cMap = solve(L, MapFac @ wtvec)
            else:
                y2cMap = solve(L, MapFac * np.tile(wtvec, (MapFac.shape[0], 1)))
        df_ = np.trace(y2cMap @ basismat)

        # Calculate SSE
        if len(dimy) < 3:
            yhat = np.array(basismat[:, :nbasis] @ coef)
            SSE = np.sum(np.square(y[:n, :] - yhat))
            if ynames is None:
                ynames = [f"rep_{i+1}" for i in range(nrep)]
        else:
            SSE = 0
            yhat = np.zeros((n, nrep, nvar))
            for ivar in range(nvar):
                yhat[:, :, ivar] = basismat[:, :nbasis] @ coef[:, :, ivar]
                SSE += np.sum(np.square(y[:n, :, ivar] - yhat[:, :, ivar]))
            if ynames is None:
                ynames = [f"rep_{i+1}" for i in range(nrep)]
            if vnames is None:
                vnames = [f"value_{i+1}" for i in range(nvar)]

        # Calculate GCV
        if df_ < n:
            if len(dimy) < 3:
                gcv = np.zeros(nrep)
                for i in range(nrep):
                    SSEi = np.sum(np.square((y[:n, i] - yhat[:, i])))
                    gcv[i] = (SSEi / n) / np.square((n - df_) / n)
            else:
                gcv = np.zeros((nrep, nvar))
                for ivar in range(nvar):
                    for i in range(nrep):
                        SSEi = np.sum(
                            np.square(y[:n, i, ivar] - yhat[:, i, ivar])
                        )
                        gcv[i, ivar] = (SSEi / n) / np.square((n - df_) / n)
                gcv = np.array(gcv).tolist()
        else:
            gcv = None
        if fdnames is None:
            fdnames = {
                "time": (
                    tnames.tolist()
                    if tnames is not None
                    else [f"time_{i+1}" for i in range(n)]
                ),
                "reps": ynames,
                "values": vnames,
            }
        if len(dimy) < 3:
            coef = np.asmatrix(coef)
            fdobj = fd(coef[:nbasis, :], basisobj, fdnames)
        else:
            fdobj = fd(coef[:nbasis, :, :], basisobj, fdnames)
        if penmat is not None and covariates is not None:
            penmat = penmat[:nbasis, :nbasis]

        # Return results
        smoothlist = {
            "fd": fdobj,
            "df": df_,
            "gcv": gcv,
            "beta": beta_,
            "SSE": SSE,
            "penmat": penmat,
            "y2cMap": y2cMap,
            "argvals": argvals,
            "y": y0,
        }
        return smoothlist
