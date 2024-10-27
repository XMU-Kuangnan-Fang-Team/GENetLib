import math
import numpy as np

from GENetLib.basis_fd import basis_fd


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

