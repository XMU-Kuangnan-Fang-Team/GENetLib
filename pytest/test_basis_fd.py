import pytest
import numpy as np
from GENetLib.basis_fd import basis_fd


def test_basis_fd_default():
    result = basis_fd()
    assert result['btype'] == 'bspline'
    assert result['rangeval'] == [0, 1]
    assert result['nbasis'] == 2

def test_basis_fd_custom():
    result = basis_fd(btype="bspline", rangeval=[0, 10], nbasis=3, params=[1, 2, 3])
    assert result['btype'] == 'bspline'
    assert result['rangeval'] == [0, 10]
    assert result['nbasis'] == 3

def test_basis_fd_expon():
    result = basis_fd(btype="expon", nbasis=2, params=[0.5, 1.5])
    assert result['btype'] == 'expon'
    assert result['nbasis'] == 2

def test_basis_fd_fourier():
    result = basis_fd(btype="fourier", nbasis=4, params=[2*np.pi])
    assert result['btype'] == 'fourier'
    assert result['nbasis'] == 5  # Fourier basis with even nbasis should increment by 1

def test_basis_fd_errors():
    with pytest.raises(ValueError):
        basis_fd(nbasis=-1)  # Negative nbasis
    with pytest.raises(ValueError):
        basis_fd(nbasis=3.5)  # Non-integer nbasis
    with pytest.raises(ValueError):
        basis_fd(btype="invalid")  # Unrecognizable basis type
    with pytest.raises(ValueError):
        basis_fd(btype="fourier", params=[-1])  # Negative period for Fourier basis
