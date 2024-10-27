import pytest
import numpy as np
from GENetLib.basis_fd import basis_fd


def test_basis_fd_defaults():
    basis = basis_fd()
    assert basis['btype'] == 'bspline'
    assert basis['rangeval'] == [0, 1]
    assert basis['nbasis'] == 2

def test_basis_fd_custom_params():
    basis = basis_fd(btype="bspline", rangeval=[1, 2], nbasis=4, params=[0.5])
    assert basis['btype'] == 'bspline'
    assert basis['rangeval'] == [1, 2]
    assert basis['nbasis'] == 4

def test_basis_fd_constant():
    basis = basis_fd(btype="const")
    assert basis['btype'] == 'const'

def test_basis_fd_polynomial():
    basis = basis_fd(btype="polynomial", nbasis=3, params=[0, 1])
    assert basis['btype'] == 'polynomial'
    assert basis['params'] == [0, 1]

def test_basis_fd_fourier_valid():
    basis = basis_fd(btype="fourier", nbasis=4, params=[np.pi])
    assert basis['btype'] == 'fourier'
    assert basis['params'] == [np.pi]

def test_basis_fd_fourier_invalid_period():
    with pytest.raises(ValueError):
        basis_fd(btype="fourier", params=[-1])

def test_basis_fd_fourier_even_nbasis():
    with pytest.raises(ValueError):
        basis_fd(btype="fourier", nbasis=4)

def test_basis_fd_bspline():
    basis = basis_fd(btype="bspline", rangeval=[0, 1], nbasis=5, params=[0.25, 0.75])
    assert basis['btype'] == 'bspline'

def test_basis_fd_bspline_breaks_out_of_range():
    with pytest.raises(ValueError):
        basis_fd(btype="bspline", rangeval=[0, 1], params=[2])

def test_basis_fd_exponential():
    basis = basis_fd(btype="expon", nbasis=2, params=[0.5])
    assert basis['btype'] == 'expon'

def test_basis_fd_monomial():
    basis = basis_fd(btype="monom", nbasis=3, params=[1, 2, 3])
    assert basis['btype'] == 'monom'

def test_basis_fd_polyg():
    basis = basis_fd(btype="polygonal", nbasis=4, params=[0, 1, 2, 3])
    assert basis['btype'] == 'polygonal'

def test_basis_fd_power():
    basis = basis_fd(btype="power", nbasis=3, params=[1, 2, 3])
    assert basis['btype'] == 'power'

def test_basis_fd_unknown_basis_type():
    with pytest.raises(ValueError):
        basis_fd(btype="unknown")

def test_basis_fd_quadvals():
    quadvals = [[0, 1], [1, 0]]
    basis = basis_fd(quadvals=quadvals)
    assert basis['quadvals'] == quadvals

def test_basis_fd_quadvals_too_few():
    with pytest.raises(ValueError):
        basis_fd(quadvals=[[0]])

def test_basis_fd_quadvals_wrong_columns():
    with pytest.raises(ValueError):
        basis_fd(quadvals=[[0, 1, 2]])

def test_basis_fd_values():
    values = [[1, 2], [3, 4]]
    basis = basis_fd(values=values)
    assert basis['values'] == values

def test_basis_fd_values_mismatch_rows():
    with pytest.raises(ValueError):
        basis_fd(quadvals=[[0, 1], [1, 0]], values=[[1]])

def test_basis_fd_values_mismatch_columns():
    with pytest.raises(ValueError):
        basis_fd(nbasis=3, quadvals=[[0, 1], [1, 0]], values=[[1, 2]])

def test_basis_fd_basisvalues():
    basisvalues = [[0, 1], [2, 3]]
    basis = basis_fd(basisvalues=basisvalues)
    assert basis['basisvalues'] == basisvalues

def test_basis_fd_basisvalues_not_2d():
    with pytest.raises(ValueError):
        basis_fd(basisvalues=[1, 2])

def test_basis_fd_dropind():
    dropind = [1]
    basis = basis_fd(dropind=dropind)
    assert basis['dropind'] == dropind


def test_basis_fd_dropind_out_of_range():
    with pytest.raises(ValueError):
        basis_fd(nbasis=2, dropind=[3])

def test_basis_fd_dropind_duplicates():
    with pytest.raises(ValueError):
        basis_fd(dropind=[1, 1])

def test_basis_fd_dropind_too_many():
    with pytest.raises(ValueError):
        basis_fd(nbasis=2, dropind=[1, 2])
