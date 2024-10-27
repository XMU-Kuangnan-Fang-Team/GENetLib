import pytest
import numpy as np
from GENetLib.basis_fd import basis_fd


def test_basisfd_default_params():
    result = BasisFD()
    assert result['btype'] == 'bspline'
    assert result['rangeval'] == [0, 1]
    assert result['nbasis'] == 2
    assert result['params'] == []
    assert result['dropind'] == []
    assert result['quadvals'] == []
    assert result['values'] == []
    assert result['basisvalues'] == []


def test_basisfd_btype():
    result = BasisFD(btype="bspline")
    assert result['btype'] == 'bspline'
    result = BasisFD(btype="con")
    assert result['btype'] == 'const'


def test_basisfd_nbasis():
    result = BasisFD(nbasis=5)
    assert result['nbasis'] == 5


def test_basisfd_nbasis_invalid():
    with pytest.raises(ValueError):
        BasisFD(nbasis=0)
    with pytest.raises(ValueError):
        BasisFD(nbasis=3.5)


def test_basisfd_quadvals():
    quadvals = np.array([[0, 1], [1, 0]])
    result = BasisFD(quadvals=quadvals)
    assert np.array_equal(result['quadvals'], quadvals)


def test_basisfd_quadvals_invalid():
    with pytest.raises(ValueError):
        BasisFD(quadvals=np.array([[0]]))
    with pytest.raises(ValueError):
        BasisFD(quadvals=np.array([[0, 1, 2]]))


def test_basisfd_values():
    values = np.array([[1, 2], [3, 4]])
    result = BasisFD(values=values)
    assert np.array_equal(result['values'], values)


def test_basisfd_values_invalid():
    with pytest.raises(ValueError):
        BasisFD(values=np.array([[1]]))


def test_basisfd_basisvalues():
    basisvalues = [[0, 1], [2, 3]]
    result = BasisFD(basisvalues=basisvalues)
    assert result['basisvalues'] == basisvalues


def test_basisfd_basisvalues_invalid():
    with pytest.raises(ValueError):
        BasisFD(basisvalues=[[1]])


def test_basisfd_dropind():
    dropind = [1]
    result = BasisFD(dropind=dropind)
    assert result['dropind'] == dropind


def test_basisfd_dropind_invalid():
    with pytest.raises(ValueError):
        BasisFD(dropind=[1, 1])
    with pytest.raises(ValueError):
        BasisFD(dropind=[10])


def test_basisfd_fourier():
    params = [2]
    result = BasisFD(btype="fourier", params=params, nbasis=4)
    assert result['params'] == params


def test_basisfd_bspline():
    params = [0.5, 1.5]
    result = BasisFD(btype="bspline", params=params, rangeval=[0, 2], nbasis=3)
    assert result['params'] == params


def test_basisfd_unknown_btype():
    with pytest.raises(ValueError):
        BasisFD(btype="unknown")
