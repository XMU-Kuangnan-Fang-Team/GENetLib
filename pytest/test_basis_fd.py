import pytest
import numpy as np
from GENetLib.fda_func import basis_fd


# Test default parameters
def test_basis_fd_defaults():
    result = basis_fd()
    assert result["btype"] == "bspline"
    assert result["rangeval"] == [0, 1]
    assert result["nbasis"] == 2


# Test custom parameters
def test_basis_fd_custom():
    result = basis_fd(
        btype="bspline", rangeval=[0, 10], nbasis=3, params=[1, 2, 3]
    )
    assert result["btype"] == "bspline"
    assert result["rangeval"] == [0, 10]
    assert result["nbasis"] == 3


# Test constant basis function
def test_basis_fd_constant():
    result = basis_fd(btype="const", rangeval=[0, 1], nbasis=1)
    assert result["btype"] == "const"


# Test Fourier basis function with valid period
def test_basis_fd_fourier_valid():
    result = basis_fd(
        btype="fourier", rangeval=[0, 1], nbasis=4, params=[2 * np.pi]
    )
    assert result["btype"] == "fourier"
    assert result["params"] == [2 * np.pi]


# Test Fourier basis function with invalid period
def test_basis_fd_fourier_invalid_period():
    with pytest.raises(ValueError):
        basis_fd(btype="fourier", rangeval=[0, 1], params=[-1])


# Test Fourier basis function with even nbasis
def test_basis_fd_fourier_even_nbasis():
    result = basis_fd(
        btype="fourier", rangeval=[0, 1], nbasis=4, params=[2 * np.pi]
    )
    assert result["nbasis"] == 5  # Should automatically adjust to odd


# Test B-spline basis function
def test_basis_fd_bspline():
    result = basis_fd(
        btype="bspline", rangeval=[0, 1], nbasis=5, params=[0.25, 0.75]
    )
    assert result["btype"] == "bspline"


# Test B-spline basis function with breaks out of range
def test_basis_fd_bspline_breaks_out_of_range():
    with pytest.raises(ValueError):
        basis_fd(btype="bspline", rangeval=[0, 1], params=[2])


# Test monomial basis function
def test_basis_fd_monomial():
    result = basis_fd(
        btype="monom", rangeval=[0, 1], nbasis=3, params=[1, 2, 3]
    )
    assert result["btype"] == "monom"


# Test polynomial basis function
def test_basis_fd_polyg():
    result = basis_fd(
        btype="polygonal", rangeval=[0, 1], nbasis=4, params=[0, 1, 2, 3]
    )
    assert result["btype"] == "polygonal"


# Test power basis function
def test_basis_fd_power():
    result = basis_fd(
        btype="power", rangeval=[0, 1], nbasis=3, params=[1, 2, 3]
    )
    assert result["btype"] == "power"


# Test unknown basis type
def test_basis_fd_unknown_basis_type():
    with pytest.raises(ValueError):
        basis_fd(btype="unknown", rangeval=[0, 1], nbasis=1)


# Test quadvals parameter with too few points
def test_basis_fd_quadvals_too_few():
    with pytest.raises(ValueError):
        basis_fd(quadvals=np.array([[0]]), rangeval=[0, 1], nbasis=1)


# Test quadvals parameter with wrong number of columns
def test_basis_fd_quadvals_wrong_columns():
    with pytest.raises(ValueError):
        basis_fd(quadvals=np.array([[0, 1, 2]]), rangeval=[0, 1], nbasis=1)


# Test values parameter with mismatched rows
def test_basis_fd_values_mismatch_rows():
    with pytest.raises(ValueError):
        basis_fd(
            values=np.array([[1]]),
            quadvals=np.array([[0, 1], [1, 0]]),
            rangeval=[0, 1],
            nbasis=1,
        )


# Test values parameter with mismatched columns
def test_basis_fd_values_mismatch_columns():
    with pytest.raises(ValueError):
        basis_fd(
            values=np.array([[1, 2]]),
            quadvals=np.array([[0, 1], [1, 0]]),
            rangeval=[0, 1],
            nbasis=3,
        )


# Test basisvalues parameter not 2D
def test_basis_fd_basisvalues_not_2d():
    with pytest.raises(ValueError):
        basis_fd(basisvalues=[1, 2], rangeval=[0, 1], nbasis=1)


# Test dropind parameter out of range
def test_basis_fd_dropind_out_of_range():
    with pytest.raises(ValueError):
        basis_fd(dropind=[3], rangeval=[0, 1], nbasis=2)


# Test dropind parameter duplicates
def test_basis_fd_dropind_duplicates():
    with pytest.raises(ValueError):
        basis_fd(dropind=[1, 1], rangeval=[0, 1], nbasis=2)


# Test dropind parameter too many
def test_basis_fd_dropind_too_many():
    with pytest.raises(ValueError):
        basis_fd(dropind=[1, 2], rangeval=[0, 1], nbasis=2)
