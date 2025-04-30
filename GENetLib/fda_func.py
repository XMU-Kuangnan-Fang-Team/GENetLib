import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.linalg import solve
import matplotlib.pyplot as plt


'''Format for functional data'''

def basis_fd(btype=None, rangeval=None, nbasis=None, params=None, dropind=None, quadvals=None, values=None, basisvalues=None):
    if btype is None and rangeval is None and nbasis is None and params is None and dropind is None and quadvals is None and values is None and basisvalues is None:
        btype = "bspline"
        rangeval = [0, 1]
        nbasis = 2
        params = []
        dropind = []
        quadvals = []
        values = []
        basisvalues = []
        basisobj = {"btype": btype, "rangeval": rangeval, "nbasis": nbasis, 
                    "params": params, "dropind": dropind, "quadvals": quadvals, 
                    "values": values, "basisvalues": basisvalues}
        return basisobj
    if btype in ["bspline", "Bspline", "spline", "Bsp", "bsp"]:
        btype = "bspline"
    elif btype in ["con", "const", "constant"]:
        btype = "const"
    elif btype in ["exp", "expon", "exponential"]:
        btype = "expon"
    elif btype in ["Fourier", "fourier", "Fou", "fou"]:
        btype = "fourier"
    elif btype in ["mon", "monom", "monomial"]:
        btype = "monom"
    elif btype in ["polyg", "polygon", "polygonal"]:
        btype = "polygonal"
    elif btype in ["polynomial", "polynom"]:
        btype = "polynomial"
    elif btype in ["pow", "power"]:
        btype = "power"
    else:
        btype = "unknown"
    if quadvals is None:
        quadvals = []
    elif len(quadvals) != 0:
        nquad, ncol = quadvals.shape
        if nquad == 2 and ncol > 2:
            quadvals = quadvals.T
            nquad, ncol = quadvals.shape
        if nquad < 2:
            raise ValueError("Less than two quadrature points are supplied.")
        if ncol != 2:
            raise ValueError("'quadvals' does not have two columns.")
    if values is not None and len(values) != 0:
        n, k = values.shape
        if n != nquad:
            raise ValueError("Number of rows in 'values' not equal to number of quadrature points.")
        if k != nbasis:
            raise ValueError("Number of columns in 'values' not equal to number of basis functions.")
    else:
        values = []
    if basisvalues is not None and len(basisvalues) != 0:
        if not isinstance(basisvalues, list):
            raise ValueError("BASISVALUES is not a list object.")
        sizevec = np.array(basisvalues).shape
        if len(sizevec) != 2:
            raise ValueError("BASISVALUES is not 2-dimensional.")
    else:
        basisvalues = []
    if dropind is None:
        dropind = []
    if len(dropind) > 0:
        ndrop = len(dropind)
        if ndrop >= nbasis:
            raise ValueError("Too many index values in DROPIND.")
        dropind = sorted(dropind)
        if ndrop > 1 and any(np.diff(dropind)) == 0:
            raise ValueError("Multiple index values in DROPIND.")
        for i in range(ndrop):
            if dropind[i] < 1 or dropind[i] > nbasis:
                raise ValueError("A DROPIND index value is out of range.")
    if btype == "fourier":
        period = params[0]
        if period <= 0:
            raise ValueError("Period must be positive for a Fourier basis")
        params = period
        if (2 * (nbasis // 2)) == nbasis:
            nbasis = nbasis + 1
    elif btype == "bspline":
        if params:
            nparams = len(params)
            if nparams > 0:
                if params[0] <= rangeval[0]:
                    raise ValueError("Smallest value in BREAKS not within RANGEVAL")
                if params[nparams-1] >= rangeval[1]:
                    raise ValueError("Largest value in BREAKS not within RANGEVAL")
    elif btype in ["expon", "polynomial", "power", "monom", "polygonal"]:
        if len(params) != nbasis:
            raise ValueError(f"No. of parameters not equal to no. of basis fns for {btype} basisobj")
    elif btype == "const":
        params = 0
    else:
        raise ValueError("Unrecognizable basis")
    if btype == "fourier":
        basisobj = {
            'btype': btype, 
            'rangeval': rangeval, 
            'nbasis': nbasis, 
            'params': [params], 
            'dropind': dropind, 
            'quadvals': quadvals, 
            'values': values, 
            'basisvalues': basisvalues
        }
        return basisobj
    else:
        basisobj = {
            'btype': btype, 
            'rangeval': rangeval, 
            'nbasis': nbasis, 
            'params': params, 
            'dropind': dropind, 
            'quadvals': quadvals, 
            'values': values, 
            'basisvalues': basisvalues
        }
        return basisobj

    
'''Create B-spline design matrix'''

def spline_design(knots, x, norder=4, outer_ok=False):
    nk = len(knots)
    if nk <= 0:
        raise ValueError("must have at least 'norder' knots")
    knots = np.sort(knots)
    degree = norder - 1
    x = np.array(x)
    need_outer = any(x < knots[degree]) or any(x > knots[nk - degree - 1])
    if not outer_ok and need_outer:
        raise ValueError("x must be within the knot range unless outer_ok = True")
    extended_knots = np.concatenate(([knots[0]] * (degree+1), knots, [knots[-1]] * (degree+1)))
    coef = np.eye(nk + degree + 1)
    spl = BSpline(extended_knots, coef, degree)
    if norder == 1:
        design_pre = spl(x)
        design_range = int((design_pre.shape[1]-nk+1) / 2)
        design = pd.DataFrame(design_pre[:,  design_range : - design_range])
        if outer_ok == False and sum(design.iloc[-1]) < 0.5 :
            m, n = design.shape
            design.loc[m-1, n-1] = 1
        return design
    else:
        design = pd.DataFrame(spl(x)[:, degree:-degree]).iloc[:, 1:-1]
        if outer_ok == False and sum(design.iloc[-1]) < 0.5 :
            m, n = design.shape
            design.loc[m-1, n] = 1
        return design


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


'''Create different types of basic functions for functional data'''

# B-spline
def create_bspline_basis(rangeval=None, nbasis=None, norder=4, breaks=None, 
                         dropind=None, quadvals=None, values=None, basisvalues=None, 
                         names=["bspl"]):
    
    btype = "bspline"
    if breaks is not None:
        Breaks = [float(b) for b in breaks]
        if min([Breaks[i+1] - Breaks[i] for i in range(len(Breaks)-1)]) < 0:
            raise ValueError("One or more breaks differences are negative.")
    if rangeval is None or len(rangeval) < 1:
        if breaks is None:
            rangeval = [0, 1]
        else:
            rangeval = [min(breaks), max(breaks)]
            if rangeval[1] - rangeval[0] == 0:
                raise ValueError("diff(range(breaks))==0; not allowed.")
    if rangeval[0] >= rangeval[1]:
        raise ValueError(f"rangeval[0] must be less than rangeval[1]; instead rangeval[0] = {rangeval[0]}", f" >= rangeval[1] = {rangeval[1]}")
    nbreaks = len(breaks) if breaks is not None else 0
    if nbasis is not None:
        if breaks is not None:
            nbreaks = len(breaks)
        else:
            breaks = list(np.linspace(rangeval[0], rangeval[1], num=nbasis - norder + 2))
            nbreaks = len(breaks)
    else:
        if breaks is None:
            nbasis = norder
        else:
            nbasis = len(breaks) + norder - 2
    if nbreaks > 2:
        params = breaks[1:(nbreaks - 1)]
    else:
        params = []
    basisobj = basis_fd(btype=btype, rangeval=rangeval, nbasis=nbasis, 
                        params=params, dropind=dropind, quadvals=quadvals, 
                        values=values, basisvalues=basisvalues)
    if len(names) == nbasis:
        basisobj['names'] = names
    else:
        if len(names) > 1:
            raise ValueError(f"length(names) = {len(names)}; must be either 1 or nbasis = {nbasis}")
        basisind = list(range(1, nbasis+1))
        new_names = []
        for name in names:
            for bi in basisind:
                new_name = f"{name}.{norder}.{bi}"
                new_names.append(new_name)
        basisobj['names'] = new_names
    return basisobj

# Exponential function
def create_expon_basis(rangeval = [0, 1], nbasis = None, ratevec = None, 
                       dropind = None, quadvals = None, values = None, basisvalues = None, 
                       names = ["exp"], axes = None):
    if nbasis is not None:
        if ratevec is None:
            ratevec = list(range(nbasis))
        else:
            if len(ratevec) != nbasis:
                raise ValueError(f"length(ratevec) must equal nbasis;  length(ratevec) = {len(ratevec)}", " != ", f"nbasis = {nbasis}")
            if len(set(ratevec)) != nbasis:
                raise ValueError("ratevec contains duplicates;  not allowed.")
    type_ = "expon"
    params = ratevec
    basisobj = basis_fd(btype = type_, rangeval = rangeval, nbasis = nbasis, 
                        params = params, dropind = dropind, quadvals = quadvals, 
                        values = values, basisvalues = basisvalues)
    if len(names) == nbasis:
        basisobj['names'] = names
    else:
        if len(names) > 1:
            raise ValueError(f"length(names) = {len(names)}; must be either 1 or nbasis = {nbasis}")
        new_names = []
        for name in names:
            for i in range(nbasis):
                new_name = name + str(i)
                new_names.append(new_name)
        basisobj['names'] = new_names
    if axes is not None:
        basisobj['axes'] = axes
    return basisobj

# Fourier function
def create_fourier_basis(rangeval = [0, 1], nbasis = 3, period = None, 
                         dropind = None, quadvals = None, values = None, basisvalues = None, 
                         names = None, axes = None):

    if period == None:
        period = float(np.diff(rangeval))
    btype = "fourier"
    if period is not None and period <= 0:
        raise ValueError(f"'period' must be positive, is {period}")
    params = [period]
    basisobj = basis_fd(btype = btype, rangeval = rangeval, nbasis = nbasis, 
                        params = params, dropind = dropind, quadvals = quadvals, 
                        values = values, basisvalues = basisvalues)
    if names is None:
        Nms = ["const"]
        if nbasis > 1:
            if nbasis == 3:
                Nms += ["sin", "cos"]
            else:
                nb2 = nbasis // 2
                sinCos = [f"{trig}{i}" for trig in ["sin", "cos"] for i in range(1, nb2+1)]
                Nms += sinCos
    else:
        if len(names) != nbasis:
            raise ValueError(f"conflict between nbasis and names:  nbasis = {nbasis}", 
                             f";  length(names) = {len(names)}")
    basisobj['names'] = Nms
    if axes is not None:
        basisobj['axes'] = axes
    return basisobj

# Monomial function
def create_monomial_basis(rangeval = [0, 1], nbasis = None, exponents = None, 
                          dropind = None, quadvals = None, values = None, basisvalues = None, 
                          names = ["monomial"], axes = None):
    
    btype = "monom"
    Rangeval = np.array(rangeval, dtype=float)
    nNAr = np.isnan(Rangeval).sum()
    if nNAr > 0:
        raise ValueError(f"as.numeric(rangeval) contains {nNAr}", " NA", f";  class(rangeval) = {type(rangeval)}")
    if np.diff(Rangeval) <= 0:
        raise ValueError(f"rangeval must cover a positive range;  diff(rangeval) = {np.diff(Rangeval)}")
    if nbasis is None:
        if exponents is None:
            nbasis = 2
            exponents = [0, 1]
        else:
            if isinstance(exponents, (list, np.ndarray)):
                nbasis = len(exponents)
                if len(set(exponents)) != nbasis:
                    raise ValueError("duplicates found in exponents;  not allowed.")
            else:
                raise ValueError(f"exponents must be numeric;  class(exponents) = {type(exponents)}")
    else:
        if isinstance(nbasis, int):
            if len([nbasis]) != 1:
                raise ValueError(f"nbasis must be a scalar;  length(nbasis) = {len([nbasis])}")
            if nbasis % 1 != 0:
                raise ValueError(f"nbasis must be an integer;  nbasis%%1 = {nbasis % 1}")
            if exponents is None:
                exponents = list(range(nbasis))
            else:
                if isinstance(exponents, (list, np.ndarray)):
                    if len(exponents) != nbasis:
                        raise ValueError("length(exponents) must = nbasis;  ", 
                            f"length(exponents) = {len(exponents)}",
                            f" != nbasis = {nbasis}")
                    if len(set(exponents)) != nbasis:
                        raise ValueError("duplicates found in exponents;  not allowed.")
                    if any([i % 1 != 0 for i in exponents]):
                        raise ValueError("exponents must be integers;  some are not.")
                    if any([i < 0 for i in exponents]):
                        raise ValueError("exponents must be nonnegative;  some are not.")
                else:
                    raise ValueError(f"exponents must be numeric;  class(exponents) = {type(exponents)}")
        else:
            raise ValueError(f"nbasis must be numeric;  class(nbasis) = {type(nbasis)}")
    if dropind is None or len(dropind) == 0:
        dropind = None
    btype = "monom"
    params = exponents
    basisobj = basis_fd(btype = btype, rangeval = rangeval, nbasis = nbasis, 
                        params = params, dropind = dropind, quadvals = quadvals, 
                        values = values, basisvalues = basisvalues)
    if len(names) == nbasis:
        basisobj['names'] = names
    else:
        if len(names) > 1:
            raise ValueError(f"length(names) = {len(names)}; must be either 1 or nbasis = {nbasis}")
        new_names = []
        for name in names:
            for i in range(nbasis):
                new_name = name + str(i)
                new_names.append(new_name)
        basisobj['names'] = new_names
    if axes is not None:
        basisobj['axes'] = axes
    return basisobj

# Power function
def create_power_basis(rangeval = [0, 1], nbasis = None, exponents = None, 
                       dropind = None, quadvals = None, values = None, basisvalues = None, 
                       names = ["power"], axes = None):
    if nbasis is None:
        if exponents is None:
            nbasis = 2
            exponents = [0, 1]
        else:
            if isinstance(exponents, (list, np.ndarray)):
                nbasis = len(exponents)
                if len(set(exponents)) != nbasis:
                    raise ValueError("duplicates found in exponents;  not allowed.")
            else:
                raise ValueError(f"exponents must be numeric;  class(exponents) = {type(exponents)}")
    else:
        if isinstance(nbasis, int):
            if len([nbasis]) != 1:
                raise ValueError(f"nbasis must be a scalar;  length(nbasis) = {len(nbasis)}")
            if nbasis % 1 != 0:
                raise ValueError(f"nbasis just be an integer;  nbasis%%1 = {nbasis % 1}")
            if exponents is None:
                exponents = list(range(nbasis))
            else:
                if isinstance(exponents, (list, np.ndarray)):
                    if len(exponents) != nbasis:
                        raise ValueError(f"length(exponents) must = nbasis;  length(exponents) = {len(exponents)} != nbasis = {nbasis}")
                    if len(set(exponents)) != nbasis:
                        raise ValueError("duplicates found in exponents;  not allowed.")
                else:
                    raise ValueError(f"exponents must be numeric;  class(exponents) = {type(exponents)}")
        else:
            raise ValueError(f"nbasis must be numeric;  class(nbasis) = {type(nbasis)}")
    if dropind is None or len(dropind) == 0:
        dropind = None
    btype = "power"
    params = sorted(list(exponents))
    basisobj = basis_fd(btype = btype, rangeval = rangeval, nbasis = nbasis, 
                        params = params, dropind = dropind, quadvals = quadvals, 
                        values = values, basisvalues = basisvalues)
    if len(names) == nbasis:
        basisobj['names'] = names
    else:
        if len(names) > 1:
            raise ValueError(f"length(names) = {len(names)}; must be either 1 or nbasis = {nbasis}")
        new_names = []
        for name in names:
            for i in range(nbasis):
                new_name = name + str(i)
                new_names.append(new_name)
        basisobj['names'] = new_names
    if axes is not None:
        basisobj['axes'] = axes
    return basisobj

# Constant value
def create_constant_basis(rangeval=[0, 1], names="const", axes=None):
    btype = "const"
    nbasis = 1
    params = []
    dropind = []
    quadvals = []
    values = []
    basisvalues = []
    basisobj = basis_fd(btype=btype, rangeval=rangeval, nbasis=nbasis, 
                        params=params, dropind=dropind, quadvals=quadvals, 
                        values=values, basisvalues=basisvalues)
    basisobj['names'] = names
    if axes is not None:
        basisobj['axes'] = axes
    return basisobj


'''Convert discrete observations into functional data'''

def dense_to_func(location, X, btype, nbasis, params, Plot = False):

    if not isinstance(location, list):
        raise ValueError("location should be of list type.")
    if btype == "Bspline":
        fbasis = create_bspline_basis(rangeval = [np.min(location), np.max(location)], nbasis = nbasis, norder = params)
    if btype == "Exponential":
        fbasis = create_expon_basis(rangeval = [np.min(location), np.max(location)], nbasis = nbasis, ratevec = params)
    if btype == "Fourier":
        fbasis = create_fourier_basis(rangeval = [np.min(location), np.max(location)], nbasis = nbasis, period = params)
    if btype == "Monomial":
        fbasis = create_monomial_basis(rangeval = [np.min(location), np.max(location)], nbasis = nbasis, exponents = params)
    if btype == "Power":
        fbasis = create_power_basis(rangeval = [np.min(location), np.max(location)], nbasis = nbasis, exponents = params)
    n, m = X.shape
    truelengths = np.count_nonzero(~np.isnan(X))
    if truelengths == n * m:
        basisMatrix = get_basis_matrix(evalarg = location, basisobj = fbasis, nderiv = 0, returnMatrix = False)
        basisCoef = solve(basisMatrix.T @ basisMatrix, basisMatrix.T @ X.T)
    tofunc = fd(coef = basisCoef, basisobj = fbasis)
    if Plot:
        plt.plot(location, eval_fd(location, tofunc))
    return tofunc


'''Calculate the value of basis functions and functional objects'''

def lfd(nderiv = 0, bwtlist = None):
    if not isinstance(nderiv, int):
        raise ValueError("Order of operator is not numeric.")
    if nderiv < 0:
        raise ValueError("Order of operator is negative.")
    if bwtlist == None:
        bwtlist = [None] * nderiv
        if nderiv > 0:
            conbasis = create_constant_basis()
            bwtlist = [fd([0], conbasis) for _ in range(nderiv)]
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
    if fdnames is None:
        if ndim == 1:
            fdnames = ["time", "reps", "values"]
        if ndim == 2:
            fdnames1 = ["reps"+str(i+1) for i in range(nrep)]
            fdnames = ["time"] + [fdnames1] + ["values"]
        fdnames = dict(zip(["args", "reps", "funs"], fdnames))
    fdobj = {"coefs": coef, "basis": basisobj, "fdnames": fdnames}
    return fdobj


def fd_chk(fdobj):
    if 'coefs' in fdobj.keys():
        coef = fdobj['coefs']
    else:
        coef = np.diag([1]*(fdobj['nbasis'] - len(fdobj['dropind'])))
        fdobj = fd(coef, fdobj)
    coef = pd.DataFrame(np.array(coef))
    coefd = coef.shape
    if len(coefd) > 2:
        raise ValueError("Functional data object must be univariate")
    nrep = coefd[1]
    return [nrep, fdobj]


def knotmultchk(basisobj, knotmult):
    btype = basisobj['btype']
    if btype == "bspline":
        params = basisobj['params']
        nparams = len(params)
        norder = basisobj['nbasis'] - nparams
        if norder == 1:
            knotmult.extend(params)
        else:
            if nparams > 1:
                for i in range(1, nparams):
                    if params[i] == params[i-1]:
                        knotmult.append(params[i])
    return knotmult


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

