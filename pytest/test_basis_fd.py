import pytest
import numpy as np
from GENetLib.basis_fd import basis_fd

def test_basisfd_default():
    basisobj = basis_fd()
    assert basisobj['btype'] == "bspline"
    assert basisobj['rangeval'] == [0, 1]
    assert basisobj['nbasis'] == 2
    assert basisobj['params'] == []
    assert basisobj['dropind'] == []
    assert basisobj['quadvals'] == []
    assert basisobj['values'] == []
    assert basisobj['basisvalues'] == []

def test_basisfd_custom():
    btype = "bspline"
    rangeval = [0, 2]
    nbasis = 3
    params = [0.5]
    dropind = [2]
    quadvals = np.array([[0, 0], [1, 0], [2, 0]])
    values = np.array([[1, 2, 3], [4, 5, 6]])
    basisobj = basis_fd(btype, rangeval, nbasis, params, dropind, quadvals, values)
    assert basisobj['btype'] == btype
    assert basisobj['rangeval'] == rangeval
    assert basisobj['nbasis'] == nbasis
    assert basisobj['params'] == params
    assert basisobj['dropind'] == dropind
    assert np.array_equal(basisobj['quadvals'], quadvals)
    assert np.array_equal(basisobj['values'], np.delete(values, dropind, axis=1))

def test_basisfd_invalid_btype():
    with pytest.raises(ValueError):
        basis_fd(btype="invalid")

def test_basisfd_invalid_nbasis():
    with pytest.raises(ValueError):
        basis_fd(nbasis=-1)

def test_basisfd_invalid_quadvals():
    quadvals = np.array([[0], [1], [2]])
    with pytest.raises(ValueError):
        basis_fd(quadvals=quadvals)

def test_basisfd_invalid_values():
    values = np.array([[1, 2], [3, 4]])
    quadvals = np.array([[0, 0], [1, 0]])
    with pytest.raises(ValueError):
        basis_fd(values=values, quadvals=quadvals)

def test_basisfd_invalid_dropind():
    dropind = [3]
    with pytest.raises(ValueError):
        basis_fd(dropind=dropind, nbasis=2)
