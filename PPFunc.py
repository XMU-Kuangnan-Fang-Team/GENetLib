import numpy as np
import pandas as pd

def ppBspline(t):
    norder = len(t) - 1
    ncoef = 2 * (norder - 1)
    def outer_product(a, b):
        a = np.array(a)[:, np.newaxis]
        b = np.array(b)[np.newaxis, :]
        outer = a * b
        return outer
    if norder > 1:
        adds = [1] * (norder - 1)
        tt = np.concatenate(([i * t[0] for i in adds], t, [i * t[-1] for i in adds]))
        gapin = np.nonzero(np.diff(tt) > 0)[0] + 1
        ngap = len(gapin)
        iseq = list(range(2 - norder, norder))
        ind = outer_product([1]*ngap, iseq) + outer_product(gapin, [1]*ncoef)
        if len(ind) == 0:
           Coeff = np.full((1, norder), np.nan)
           index = 0
        else:
            tx = np.array(tt)[ind-1]
            ty = tx - outer_product(tt[gapin-1], [1]*ncoef)
            b = outer_product([1]*ngap, list(range(1 - norder, 1))) + outer_product(gapin, [1]*norder)
            a = np.concatenate(([i * 0 for i in adds], [1], [i * 0 for i in adds]))
            d = np.array(a)[b-1]
            d = pd.DataFrame(d)
            for j in range(1, norder):
                for i in range(norder - j):
                    ind1 = i + norder - 1
                    ind2 = i + j - 1
                    d.loc[:, i] = (ty[:, ind1] * d.loc[:, i] - ty[:, ind2] * d.loc[:, i + 1]) / (ty[:, ind1] - ty[:, ind2])
            Coeff = d
            for j in range(2, norder + 1):
                factor = (norder - j + 1) / (j - 1)
                ind = range(norder - 1, j - 2, -1)
                for i in ind:
                    Coeff.iloc[:, i] = factor * (Coeff.iloc[:, i] - Coeff.iloc[:, i - 1]) / ty[:, i + norder - j]
            ind = list(range(norder, 0, -1))
            if ngap > 1:
                Coeff = Coeff.iloc[:, [i-1 for i in ind]]
            else:
                Coeff = Coeff.iloc[:, [i-1 for i in ind]].to_numpy().reshape(1,-1)
            index = np.array(gapin) - (norder - 1)
    else:
        Coeff = np.array([[1]])
        index = np.array([[1]])
    return [Coeff, index]

def ppDeriv(Coeff, Deriv=0):
    if np.all(np.isnan(Coeff)) == True:
        return pd.DataFrame([[len(Coeff)], [0]])
    m = Coeff.shape[0]
    k = Coeff.shape[1]  
    if Deriv < 1:
        CoeffD = pd.DataFrame(Coeff)
        return CoeffD
    if (k-Deriv) < 1:
        CoeffD = pd.DataFrame(np.zeros((m,1)))
        return CoeffD
    else:
        CoeffD = Coeff.iloc[:,0:(k-Deriv)]
        for j in range(1, k-1):
            bound1 = max(1, j-Deriv+1) 
            bound2 = min(j, k-Deriv) 
            CoeffD.iloc[:,bound1-1:bound2] = (k-j)*CoeffD.iloc[:,bound1-1:bound2] 
        return CoeffD

