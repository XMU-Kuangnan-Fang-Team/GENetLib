import math
import numpy as np
from BasisFD import BasisFD


def create_bspline_basis(rangeval=None, nbasis=None, norder=4, breaks=None, 
                         dropind=None, quadvals=None, values=None, basisvalues=None, 
                         names=["bspl"]):
    btype = "bspline"
    if breaks is not None:
        Breaks = [float(b) for b in breaks]
        if min([Breaks[i+1] - Breaks[i] for i in range(len(Breaks)-1)]) < 0:
            raise ValueError("One or more breaks differences are negative.")
        if any([math.isnan(b) for b in Breaks]):
            raise ValueError("breaks contains NAs; not allowed.")
        if any([math.isinf(b) for b in Breaks]):
            raise ValueError("breaks contains Infs; not allowed.")
    if rangeval is None or len(rangeval) < 1:
        if breaks is None:
            rangeval = [0, 1]
        else:
            rangeval = [min(breaks), max(breaks)]
            if rangeval[1] - rangeval[0] == 0:
                raise ValueError("diff(range(breaks))==0; not allowed.")
    if len(rangeval) == 1:
        if rangeval[0] <= 0:
            raise ValueError(f"'rangeval' a single value that is not positive, is {rangeval}")
        rangeval = [0, rangeval[0]]
    if len(rangeval) > 2:
        if breaks is not None:
            raise ValueError(f"breaks can not be provided with length(rangeval)>2; length(rangeval) = {len(rangeval)}",  f" and length(breaks) = {len(breaks)}")
        breaks = rangeval
        rangeval = [min(breaks), max(breaks)]
    if rangeval[0] >= rangeval[1]:
        raise ValueError(f"rangeval[0] must be less than rangeval[1]; instead rangeval[0] = {rangeval[0]}", f" >= rangeval[1] = {rangeval[1]}")
    if not isinstance(norder, int):
        raise ValueError(f"norder must be numeric; class(norder) = {type(norder)}")
    if norder <= 0:
        raise ValueError(f"norder must be positive, is {norder}")
    if norder % 1 > 0:
        raise ValueError(f"norder must be an integer, = {norder}", f", with fractional part = {norder % 1}")
    nbreaks = len(breaks) if breaks is not None else 0
    if nbasis is not None:
        if not isinstance(nbasis, int):
            raise ValueError(f"nbasis must be numeric, is {type(nbasis)}")
        if nbasis < 1:
            raise ValueError(f"nbasis must be a single positive integer; nbasis = {nbasis}")
        if nbasis % 1 > 0:
            raise ValueError(f"nbasis is not an integer, = {nbasis}", f", with fractional part = {nbasis % 1}")
        if nbasis < norder:
            raise ValueError(f"nbasis must be at least norder; nbasis = {nbasis}", f"; norder = {norder}")
        if breaks is not None:
            nbreaks = len(breaks)
            if nbreaks < 2:
                raise ValueError("Number of values in argument 'breaks' less than 2.")
            if breaks[0] != rangeval[0] or breaks[nbreaks-1] != rangeval[1]:
                raise ValueError("Range of argument 'breaks' not identical to that of argument 'rangeval'.")
            if min([breaks[i+1] - breaks[i] for i in range(nbreaks-1)]) < 0:
                raise ValueError("Values in argument 'breaks' are decreasing.")
            if nbasis != norder + nbreaks - 2:
                raise ValueError(f"Relation nbasis = norder + length(breaks) - 2 does not hold; nbasis = {nbasis}", f"norder = {norder}", "length(breaks) = {len(breaks)}")
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
    basisobj = BasisFD(btype=btype, rangeval=rangeval, nbasis=nbasis, 
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

def create_expon_basis(rangeval = [0, 1], nbasis = None, ratevec = None, 
                       dropind = None, quadvals = None, values = None, basisvalues = None, 
                       names = ["exp"], axes = None):
    if not isinstance(rangeval, (list, np.ndarray)):
        raise ValueError(f"rangeval must be numeric;  class(rangeval) = {type(rangeval)}")
    if len(rangeval) < 1:
        raise ValueError("rangeval must be a numeric vector of length 2;  ", "length(rangeval) = 0.")
    if len(rangeval) == 1:
        if rangeval[0] <= 0:
            raise ValueError(f"rangeval a single value that is not positive:  {rangeval}")
        rangeval = [0, rangeval[0]]
    if len(rangeval) > 2:
        raise ValueError("rangeval must be a vector of length 2;  ", f"length(rangeval) = {len(rangeval)}")
    if np.diff(rangeval) <= 0:
        raise ValueError(f"rangeval must cover a positive range;  diff(rangeval) = {np.diff(rangeval)}")
    if nbasis is None:
        if ratevec is None:
            nbasis = 2
            ratevec = [0, 1]
        else:
            nbasis = len(ratevec)
            if nbasis < 1:
                raise ValueError("ratevec must have positive length;  length(ratevec) = 0")
            if not isinstance(ratevec, (list, np.ndarray)):
                raise ValueError(f"ratevec must be numeric;  class(ratevec) = {type(ratevec)}")
            if len(set(ratevec)) != nbasis:
                raise ValueError("ratevec contains duplicates;  not allowed.")
    else:
        if ratevec is None:
            ratevec = list(range(nbasis))
        else:
            if len(ratevec) != nbasis:
                raise ValueError(f"length(ratevec) must equal nbasis;  length(ratevec) = {len(ratevec)}", " != ", f"nbasis = {nbasis}")
            if len(set(ratevec)) != nbasis:
                raise ValueError("ratevec contains duplicates;  not allowed.")
    if dropind is not None and len(dropind) > 0:
        if not isinstance(dropind, (list, np.ndarray)):
            raise ValueError(f"dropind must be numeric;  is {type(dropind)}")
        doops = [i for i in dropind if i % 1 > 0]
        if len(doops) > 0:
            raise ValueError(f"dropind must be integer;  element {doops[0]}",  
                f" = {dropind[doops[0]]}", "; fractional part = {dropind[doops[0]] % 1}")
        doops0 = [i for i in dropind if i <= 0]
        if len(doops0) > 0:
            raise ValueError(f"dropind must be positive integers;  element {doops0[0]}", 
                             f" = {dropind[doops0[0]]}",  " is not.")
        doops2 = [i for i in dropind if i > nbasis]
        if len(doops2) > 0:
            raise ValueError(f"dropind must not exceed nbasis = {nbasis}",
                f";  dropind[{doops2[0]}", f"] = {dropind[doops2[0]]}")
        dropind = sorted(dropind)
        if len(dropind) > 1:
            if min(np.diff(dropind)) == 0:
                raise ValueError("Multiple index values in DROPIND.")
    type_ = "expon"
    params = ratevec
    basisobj = BasisFD(btype = type_, rangeval = rangeval, nbasis = nbasis, 
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

def create_fourier_basis(rangeval = [0, 1], nbasis = 3, period = None, 
                         dropind = None, quadvals = None, values = None, basisvalues = None, 
                         names = None, axes = None):
    if period == None:
        period = float(np.diff(rangeval))
    btype = "fourier"
    if len(rangeval) < 1:
        raise ValueError("length(rangeval) = 0;  not allowed.")
    if len(rangeval) == 1:
        if rangeval[0] <= 0:
            raise ValueError("RANGEVAL a single value that is not positive.")
        rangeval = [0, rangeval[0]]
    if period is not None and not isinstance(period, (int, float)):
        raise ValueError(f"period must be numeric;  class(period) = {type(period)}")
    if period is not None and period <= 0:
        raise ValueError(f"'period' must be positive, is {period}")
    if not isinstance(nbasis, int):
        raise ValueError(f"nbasis must be numeric;  class(nbasis) = {type(nbasis)}")
    if nbasis <= 0:
        raise ValueError(f"nbasis must be positive;  is {nbasis}")
    if dropind is not None and len(dropind) > 0:
        if len(dropind) >= nbasis:
            raise ValueError("dropind request deleting more basis functions than exist.")
        dropind = sorted(dropind)
        if any([i % 1 > 0 for i in dropind]):
            raise ValueError("some dropind are not integers.")
        dropind = [round(i) for i in dropind]
        if len(dropind) > 1:
            if min(np.diff(dropind)) == 0:
                raise ValueError("dropind requists deleting the same basis function more than once.")
        for i in dropind:
            if i < 1 or i > nbasis:
                raise ValueError(f"dropind contains an index value out of range:  {i}")
    params = [period]
    basisobj = BasisFD(btype = btype, rangeval = rangeval, nbasis = nbasis, 
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

def create_monomial_basis(rangeval = [0, 1], nbasis = None, exponents = None, 
                          dropind = None, quadvals = None, values = None, basisvalues = None, 
                          names = ["monomial"], axes = None):
    btype = "monom"
    Rangeval = np.array(rangeval, dtype=float)
    if len(rangeval) < 1:
        raise ValueError("rangeval must be a numeric vector of length 2; length(rangeval) = 0.")
    if len(rangeval) == 1:
        if rangeval[0] <= 0:
            raise ValueError(f"rangeval a single value that is not positive:  {rangeval}")
        rangeval = [0, rangeval[0]]
    if len(rangeval) > 2:
        raise ValueError("rangeval must be a vector of length 2;  ", f"length(rangeval) = {len(rangeval)}")
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
    if dropind is not None and len(dropind) > 0:
        if not isinstance(dropind, (list, np.ndarray)):
            raise ValueError(f"dropind must be numeric;  is {type(dropind)}")
        doops = [i for i in dropind if i % 1 > 0]
        if len(doops) > 0:
            raise ValueError(f"dropind must be integer;  element {doops[0]}", 
                f" = {dropind[doops[0]]}", f"; fractional part = {dropind[doops[0]] % 1}")
        doops0 = [i for i in dropind if i <= 0]
        if len(doops0) > 0:
            raise ValueError(f"dropind must be positive integers;  element {doops0[0]}",
                             " = {dropind[doops0[0]]}", " is not.")
        doops2 = [i for i in dropind if i > nbasis]
        if len(doops2) > 0:
            raise ValueError(f"dropind must not exceed nbasis = {nbasis}", 
                             f";  dropind[{doops2[0]}", f"] = {dropind[doops2[0]]}")
        dropind = sorted(dropind)
        if len(dropind) > 1:
            if min(np.diff(dropind)) == 0:
                raise ValueError("Multiple index values in DROPIND.")
    btype = "monom"
    params = exponents
    basisobj = BasisFD(btype = btype, rangeval = rangeval, nbasis = nbasis, 
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

def create_power_basis(rangeval = [0, 1], nbasis = None, exponents = None, 
                       dropind = None, quadvals = None, values = None, basisvalues = None, 
                       names = ["power"], axes = None):
    if not isinstance(rangeval, (list, np.ndarray)):
        raise ValueError(f"rangaval must be numeric;  class(rangeval) = {type(rangeval)}")
    if len(rangeval) < 1:
        raise ValueError("rangeval must be a numeric vector of length 2;  length(rangeval) = 0.")
    if len(rangeval) == 1:
        if rangeval[0] <= 0:
            raise ValueError(f"rangeval a single value that is not positive:  {rangeval}")
        rangeval = [0, rangeval[0]]
    if len(rangeval) > 2:
        raise ValueError(f"rangeval must be a vector of length 2;  length(rangeval) = {len(rangeval)}")
    if np.diff(rangeval) <= 0:
        raise ValueError(f"rangeval must cover a positive range;  diff(rangeval) = {np.diff(rangeval)}")
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
    if any([i < 0 for i in exponents]) and rangeval[1] <= 0:
        raise ValueError("An exponent is negative and range contains 0 or negative values.")
    if dropind is None or len(dropind) == 0:
        dropind = None
    if dropind is not None and len(dropind) > 0:
        if not isinstance(dropind, (list, np.ndarray)):
            raise ValueError(f"dropind must be numeric;  is {type(dropind)}")
        doops = [i for i in dropind if i % 1 > 0]
        if len(doops) > 0:
            raise ValueError(f"dropind must be integer;  element {doops[0]} = {dropind[doops[0]]}; fractional part = {dropind[doops[0]] % 1}")
        doops0 = [i for i in dropind if i <= 0]
        if len(doops0) > 0:
            raise ValueError(f"dropind must be positive integers;  element {doops0[0]} = {dropind[doops0[0]]} is not.")
        doops2 = [i for i in dropind if i > nbasis]
        if len(doops2) > 0:
            raise ValueError(f"dropind must not exceed nbasis = {nbasis};  dropind[{doops2[0]}] = {dropind[doops2[0]]}")
        dropind = sorted(dropind)
        if len(dropind) > 1:
            if min(np.diff(dropind)) == 0:
                raise ValueError("Multiple index values in DROPIND.")
    btype = "power"
    params = sorted(list(exponents))
    basisobj = BasisFD(btype = btype, rangeval = rangeval, nbasis = nbasis, 
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

def create_constant_basis(rangeval=[0, 1], names="const", axes=None):
    if len(rangeval) == 1:
        if rangeval[0] <= 0:
            raise ValueError("RANGEVAL is a single value that is not positive.")
        rangeval = [0, rangeval[0]]
    btype = "const"
    nbasis = 1
    params = []
    dropind = []
    quadvals = []
    values = []
    basisvalues = []
    basisobj = BasisFD(btype=btype, rangeval=rangeval, nbasis=nbasis, 
                       params=params, dropind=dropind, quadvals=quadvals, 
                       values=values, basisvalues=basisvalues)
    basisobj['names'] = names
    if axes is not None:
        basisobj['axes'] = axes
    return basisobj

