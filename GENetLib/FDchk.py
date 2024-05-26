import numpy as np
import pandas as pd
from FD import FD


def FDchk(fdobj):
    if 'coefs' in fdobj.keys():
        coef = fdobj['coefs']
    else:
        coef = np.diag([1]*(fdobj['nbasis'] - len(fdobj['dropind'])))
        fdobj = FD(coef, fdobj)
    coef = pd.DataFrame(np.array(coef))
    coefd = coef.shape
    if len(coefd) > 2:
        raise ValueError("Functional data object must be univariate")
    nrep = coefd[1]
    return [nrep, fdobj]

