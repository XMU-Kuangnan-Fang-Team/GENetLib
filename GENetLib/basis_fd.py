import numpy as np


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

    
