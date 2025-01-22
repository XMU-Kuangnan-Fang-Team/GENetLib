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

