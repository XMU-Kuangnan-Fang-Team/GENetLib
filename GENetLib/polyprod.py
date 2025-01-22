import numpy as np


def polyprod(Coeff1, Coeff2):
    polyorder1 = Coeff1.shape[0]
    norder1 = Coeff1.shape[1]
    polyorder2 = Coeff2.shape[0]
    norder2 = Coeff2.shape[1]
    ndegree1 = polyorder1 - 1
    ndegree2 = polyorder2 - 1
    if ndegree1 != ndegree2:
        if ndegree1 > ndegree2:
            Coeff2 = np.vstack((Coeff2, np.zeros((ndegree1-ndegree2, norder2))))
        else:
            Coeff1 = np.vstack((Coeff1, np.zeros((ndegree2-ndegree1, norder1))))
    D = max(ndegree1, ndegree2)
    N = 2*D + 1
    convmat = np.zeros((norder1, norder2, N))
    for i in range(D):
        ind = np.arange(i+1)
        if len(ind) == 1:
            convmat[:, :, i+1] = np.outer(Coeff1[ind, :], Coeff2[i-ind+1, :])
            convmat[:, :, N-i-1] = np.outer(Coeff1[D-ind, :], Coeff2[D-i+ind, :])
        else:
            convmat[:, :, i+1] = np.dot(Coeff1[ind, :].T, Coeff2[i-ind+1, :])
            convmat[:, :, N-i-1] = np.dot(Coeff1[D-ind, :].T, Coeff2[D-i+ind, :])
    ind = np.arange(D+1)
    convmat[:, :, D] = np.dot(Coeff1[ind, :].T, Coeff2[D-ind, :])
    if ndegree1 != ndegree2:
        convmat = convmat[:, :, :ndegree1+ndegree2+1]
        convmat = convmat.reshape((norder1, norder2, ndegree1+ndegree2+1))
    return convmat

