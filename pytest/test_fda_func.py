import numpy as np
from GENetLib.fda_func import ycheck, ppbspline, wtcheck, vec2lfd, norder_bspline, fdpar, fdparcheck, create_bspline_basis, create_fourier_basis
import pytest

def test_ycheck_valid_2d():
    y = [[1, 2, 3],
         [4, 5, 6]]
    n = 2
    result = ycheck(y, n)
    assert isinstance(result['y'], np.ndarray)
    assert result['y'].shape == (2, 3)
    assert result['ncurve'] == 3
    assert result['nvar'] == 1
    assert result['ndim'] == 2

def test_ycheck_valid_3d():
    y = np.array([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
    n = 2
    result = ycheck(y, n)
    assert result['y'].shape == (2, 2, 2)
    assert result['ncurve'] == 2
    assert result['nvar'] == 2
    assert result['ndim'] == 3

def test_ycheck_invalid_length():
    y = np.array([[1, 2, 3]])
    n = 2
    with pytest.raises(ValueError, match="Y is not the same length as ARGVALS"):
        ycheck(y, n)

def test_ycheck_invalid_type():
    y = "some invalid type"
    n = 1
    with pytest.raises(ValueError, match="Y is not of class matrix or class array."):
        ycheck(y, n)

def test_ycheck_invalid_dimensions():
    y = np.array([1, 2, 3])
    n = 3
    with pytest.raises(ValueError, match="Second argument must not have more than 3 dimensions"):
        ycheck(y, n)

def test_ppbspline_single():
    t = np.array([0])
    Coeff, index = ppbspline(t)
    np.testing.assert_array_equal(Coeff, np.array([[1]]))
    np.testing.assert_array_equal(index, np.array([[1]]))

def test_ppbspline_two():
    t = np.array([0, 1])
    Coeff, index = ppbspline(t)
    np.testing.assert_array_equal(Coeff, np.array([[1]]))
    np.testing.assert_array_equal(index, np.array([[1]]))

def test_ppbspline_three():
    t = np.array([0, 0.5, 1])
    Coeff, index = ppbspline(t)
    expected_Coeff = np.array([[2, 0], [-2, 1]])
    expected_index = np.array([1, 2])
    np.testing.assert_allclose(Coeff, expected_Coeff, rtol=1e-5, atol=1e-8)
    np.testing.assert_array_equal(index, expected_index)

def test_ppbspline_four():
    t = np.array([0, 0.25, 0.5, 0.75, 1])
    Coeff, index = ppbspline(t)
    expected_index = np.array([1, 2, 3, 4])
    assert Coeff.shape == (4, 4)
    np.testing.assert_array_equal(index, expected_index)

def test_ppbspline_list_input():
    t = [0, 0.5, 1]
    Coeff, index = ppbspline(t)
    expected_Coeff = np.array([[2, 0], [-2, 1]])
    expected_index = np.array([1, 2])
    np.testing.assert_allclose(Coeff, expected_Coeff, rtol=1e-5, atol=1e-8)
    np.testing.assert_array_equal(index, expected_index)

def test_wtcheck_default():
    n = 4
    result = wtcheck(n)
    expected = np.ones((n, 1))
    np.testing.assert_array_equal(result['wtvec'], expected)
    assert result['onewt'] == True
    assert result['matwt'] == False

def test_wtcheck_non_integer_n():
    with pytest.raises(ValueError, match="n is not an integer."):
        wtcheck(3.5)

def test_wtcheck_n_less_than_one():
    with pytest.raises(ValueError, match="n is less than 1."):
        wtcheck(0)

def test_wtcheck_vector_wrong_length():
    n = 3
    wt = np.array([1, 2])
    with pytest.raises(ValueError, match="WTVEC of wrong length"):
        wtcheck(n, wt)

def test_wtcheck_vector_with_nonpositive_value():
    n = 3
    wt = np.array([1, 0, 3])
    with pytest.raises(ValueError, match="Values in WTVEC are not positive."):
        wtcheck(n, wt)

def test_wtcheck_vector_with_NaN():
    n = 3
    wt = np.array([1, np.nan, 3])
    with pytest.raises(ValueError, match="WTVEC has NA values."):
        wtcheck(n, wt)

def test_wtcheck_scalar_wtvec():
    n = 4
    wt = np.array([3])
    result = wtcheck(n, wt)
    expected = 3 * np.ones((n, 1))
    np.testing.assert_array_equal(result['wtvec'], expected)
    assert result['onewt'] == False
    assert result['matwt'] == False

def test_wtcheck_valid_matrix():
    n = 2
    wt = np.array([[2, 0], [0, 2]])
    result = wtcheck(n, wt)
    np.testing.assert_array_equal(result['wtvec'], wt)
    assert result['matwt'] == True
    assert result['onewt'] == False

def test_wtcheck_matrix_not_positive_definite():
    n = 2
    wt = np.array([[1, 0], [0, -1]])
    with pytest.raises(ValueError, match="Weight matrix is not positive definite."):
        wtcheck(n, wt)

def test_wtcheck_matrix_complex_eigenvalues():
    n = 2
    wt = np.array([[0, -1], [1, 0]])
    with pytest.raises(ValueError, match="Weight matrix has complex eigenvalues."):
        wtcheck(n, wt)

def test_wtcheck_invalid_matrix_shape():
    n = 3
    wt = np.ones((3, 2))
    with pytest.raises(ValueError, match="WTVEC is neither a vector nor a matrix of order n."):
        wtcheck(n, wt)

def test_wtcheck_valid_vector():
    n = 3
    wt = np.array([1, 2, 3])
    result = wtcheck(n, wt)
    np.testing.assert_array_equal(result['wtvec'].flatten(), wt.flatten())
    assert result['onewt'] == False
    assert result['matwt'] == False

def test_wtcheck_vector_all_ones():
    n = 3
    wt = np.array([1, 1, 1])
    result = wtcheck(n, wt)
    np.testing.assert_array_equal(result['wtvec'].flatten(), wt.flatten())
    assert result['onewt'] == True
    assert result['matwt'] == False

def test_vec2lfd_non_list():
    result = vec2lfd(3, [0, 1])
    assert result is not None

def test_vec2lfd_list():
    result = vec2lfd([10, 20, 30], [5, 15])
    assert result is not None

def test_vec2lfd_empty():
    result = vec2lfd([], [5, 10])
    assert result is not None

def test_norder_bspline():
    x = {"nbasis": 8, "params": [1, 2, 3]}
    assert norder_bspline(x) == 5
    x = {"nbasis": 6, "params": []}
    assert norder_bspline(x) == 6
    x = {"nbasis": 4, "params": [1, 2, 3, 4]}
    assert norder_bspline(x) == 0
    x = {"nbasis": 3, "params": [1, 2, 3, 4]}
    assert norder_bspline(x) == -1


def test_fdpar():
    basisobj = create_bspline_basis(rangeval = [0, 1], nbasis = 4, norder = 3)
    fdobj = fd(coef = np.zeros((4, 4)), basisobj = basisobj)
    Lfdobj = int2lfd(2)
    fdParobj = fdpar(fdobj, None, lambda_ = 0.5, estimate = True)
    assert isinstance(fdParobj, dict)
    assert 'fd' in fdParobj and fdParobj['fd'] is not None
    assert 'lfd' in fdParobj and fdParobj['lfd'] is not None
    assert 'lambda' in fdParobj and isinstance(fdParobj['lambda'], float)
    assert 'estimate' in fdParobj and isinstance(fdParobj['estimate'], bool)
    assert 'penmat' in fdParobj


def test_fdparcheck():
    basisobj = create_fourier_basis(rangeval = [0, 1], nbasis = 3)
    fdobj = fd(coef = np.zeros((3, 3)), basisobj = basisobj)
    fdParobj = fdpar(fdobj, lambda_ = 0.1)
    checked_fdParobj = fdparcheck(fdParobj)
    assert isinstance(checked_fdParobj, dict)
    assert 'fd' in checked_fdParobj and checked_fdParobj['fd'] is not None
    assert 'lfd' in checked_fdParobj and checked_fdParobj['lfd'] is not None
    assert 'lambda' in checked_fdParobj and isinstance(checked_fdParobj['lambda'], float)
    assert 'estimate' in checked_fdParobj and isinstance(checked_fdParobj['estimate'], bool)
    assert 'penmat' in checked_fdParobj
