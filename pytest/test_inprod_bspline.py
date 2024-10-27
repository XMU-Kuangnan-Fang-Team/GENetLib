import pytest
import numpy as np
import pandas as pd
from GENetLib.create_basis import create_bspline_basis
from GENetLib.fd import fd
from GENetLib.inprod_bspline import inprod_bspline

def create_test_basis(nbasis, norder):
    rangeval = [0, 1]
    breaks = np.linspace(rangeval[0], rangeval[1], nbasis + norder - 2)
    basisobj = create_bspline_basis(rangeval, nbasis, norder, breaks)
    return basisobj

def create_test_fdobj(basisobj, coef):
    fdobj = fd(coef=coef, basisobj=basisobj)
    return fdobj

def test_inprod_bspline_normal():
    nbasis = 4
    norder = 4
    basisobj1 = create_test_basis(nbasis, norder)
    basisobj2 = create_test_basis(nbasis, norder)
    coef1 = np.random.rand(nbasis)
    coef2 = np.random.rand(nbasis)
    fdobj1 = create_test_fdobj(basisobj1, coef1)
    fdobj2 = create_test_fdobj(basisobj2, coef2)
    result = inprod_bspline(fdobj1, fdobj2)
    assert result.shape == (1, 1)

def test_inprod_bspline_diff_knots():
    nbasis = 4
    norder = 4
    basisobj1 = create_test_basis(nbasis, norder)
    basisobj2 = create_test_basis(nbasis + 1, norder)
    coef1 = np.random.rand(nbasis)
    coef2 = np.random.rand(nbasis + 1)
    fdobj1 = create_test_fdobj(basisobj1, coef1)
    fdobj2 = create_test_fdobj(basisobj2, coef2)
    with pytest.raises(Exception):
        inprod_bspline(fdobj1, fdobj2)

def test_inprod_bspline_diff_range():
    nbasis = 4
    norder = 4
    basisobj1 = create_test_basis(nbasis, norder)
    basisobj2 = create_test_basis(nbasis, norder, rangeval=[1, 2])
    coef1 = np.random.rand(nbasis)
    coef2 = np.random.rand(nbasis)
    fdobj1 = create_test_fdobj(basisobj1, coef1)
    fdobj2 = create_test_fdobj(basisobj2, coef2)
    with pytest.raises(Exception):
        inprod_bspline(fdobj1, fdobj2)

def test_inprod_bspline_invalid_coef_dim():
    nbasis = 4
    norder = 4
    basisobj = create_test_basis(nbasis, norder)
    coef1 = np.random.rand(nbasis, 2) 
    coef2 = np.random.rand(nbasis)
    fdobj1 = create_test_fdobj(basisobj, coef1)
    fdobj2 = create_test_fdobj(basisobj, coef2)
    with pytest.raises(Exception):
        inprod_bspline(fdobj1, fdobj2)
