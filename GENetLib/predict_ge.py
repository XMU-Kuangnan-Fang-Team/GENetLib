import numpy as np
import pandas as pd
from GENetLib.pre_data import pre_data2
from GENetLib.fda_func import dense_to_func, create_bspline_basis, inprod


"""Make predictions for scalar_ge and func_ge"""


def predict_scalar(ge_res, y, ytype, G, E, GE=None):
    if GE == None:
        GE = np.zeros(shape=(G.shape[0], G.shape[1] * E.shape[1]))
        k = 0
        for i in range(E.shape[1]):
            for j in range(G.shape[1]):
                GE[:, k] = E[:, i] * G[:, j]
                k = k + 1
    G, y, E, GE = pre_data2(y, G, E, GE, ytype, split_type=0, ratio=[1, 0])[:4]
    if len(ge_res) == 5:
        pred = ge_res[4](G, GE, E)
    elif len(ge_res) == 6 or len(ge_res) == 8:
        pred = ge_res[5](G, GE, E)
    else:
        if len(ge_res[0]) == 5:
            pred = ge_res[0][4](G, GE, E)
        else:
            pred = ge_res[0][5](G, GE, E)
    return pred


def predict_func(
    ge_res,
    y,
    ytype,
    G,
    E,
    location,
    nbasis1=15,
    params1=4,
    Bsplines=20,
    norder1=4,
):
    if type(G) == list and type(G[0]) == dict:
        fbasis2 = create_bspline_basis(
            rangeval=[min(location), max(location)],
            nbasis=Bsplines,
            norder=norder1,
        )
        U_list = []
        for idx, item in enumerate(G):
            basisint = inprod(
                fdobj1=item["fd"]["basis"], fdobj2=fbasis2, Lfdobj1=0, Lfdobj2=0
            )
            u_val = np.dot(item["fd"]["coefs"].reshape(-1), basisint)
            U_list.append(u_val)
        U = pd.DataFrame(np.array(U_list).reshape(len(G), -1))
    else:
        funcX = dense_to_func(
            location,
            G,
            btype="Bspline",
            nbasis=nbasis1,
            params=params1,
            Plot=False,
        )
        fbasis1 = create_bspline_basis(
            rangeval=[min(location), max(location)],
            nbasis=nbasis1,
            norder=params1,
        )
        fbasis2 = create_bspline_basis(
            rangeval=[min(location), max(location)],
            nbasis=Bsplines,
            norder=norder1,
        )
        n, m = G.shape
        funcCoef = funcX["coefs"].T
        basisint = inprod(fdobj1=fbasis1, fdobj2=fbasis2, Lfdobj1=0, Lfdobj2=0)

        def funcU(i):
            return np.dot(funcCoef[i, :], basisint)

        U = pd.DataFrame(np.array([funcU(i) for i in range(n)]).reshape(n, -1))
    GE = np.zeros(shape=(U.shape[0], U.shape[1] * E.shape[1]))
    k = 0
    for i in range(E.shape[1]):
        for j in range(U.shape[1]):
            GE[:, k] = E[:, i] * U.iloc[:, j]
            k = k + 1
    U_, y_, E_, GE_ = pre_data2(
        y, U, E, GE, ytype=ytype, split_type=0, ratio=[10, 0]
    )[:4]
    if len(ge_res[0]) == 5:
        pred = ge_res[0][4](U_, GE_, E_)
    elif len(ge_res[0]) == 6 or len(ge_res[0]) == 8:
        pred = ge_res[0][5](U_, GE_, E_)
    else:
        if len(ge_res[0][0]) == 5:
            pred = ge_res[0][0][4](U_, GE_, E_)
        else:
            pred = ge_res[0][0][5](U_, GE_, E_)
    return pred
