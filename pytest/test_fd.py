import pytest
import numpy as np
from GENetLib.fd import fd
from GENetLib.create_basis import create_bspline_basis

# Test fd function with default coefficients
def test_fd_default_coef():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    fdobj = fd(basisobj=basisobj)
    assert 'coefs' in fdobj
    assert fdobj['coefs'].shape == (4,)

# Test fd function with custom coefficients
def test_fd_custom_coef():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4)
    fdobj = fd(coef=coef, basisobj=basisobj)
    assert 'coefs' in fdobj
    assert np.all(fdobj['coefs'] == coef)

# Test fd function with multidimensional coefficients
def test_fd_multidim_coef():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4, 2)
    fdobj = fd(coef=coef, basisobj=basisobj)
    assert 'coefs' in fdobj
    assert fdobj['coefs'].shape == (4, 2)

# Test fd function with invalid coefficient type
def test_fd_invalid_coef_type():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = "not a number"
    with pytest.raises(ValueError):
        fd(coef=coef, basisobj=basisobj)

# Test fd function with custom fdnames
def test_fd_custom_fdnames():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4)
    fdnames = {"args": ["time"], "reps": ["rep1"], "funs": ["value"]}
    fdobj = fd(coef=coef, basisobj=basisobj, fdnames=fdnames)
    assert 'fdnames' in fdobj
    assert fdobj['fdnames'] == fdnames

