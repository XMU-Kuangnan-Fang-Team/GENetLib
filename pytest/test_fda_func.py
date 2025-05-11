import numpy as np
from GENetLib.fda_func import ycheck, ppbspline
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
    result = wtcheck(5)
    assert 'wtvec' in result
    assert 'onewt' in result
    assert 'matwt' in result
    assert result['wtvec'].shape == (5, 1)
    assert np.all(result['wtvec'] == 1)
    assert result['onewt'] is True
    assert result['matwt'] is False

def test_wtcheck_vector_nonone():
    wt = np.array([2, 2, 2, 2, 2])
    result = wtcheck(5, wt)
    assert result['wtvec'].shape == (5, 1)
    assert np.all(result['wtvec'] == 2)
    assert result['onewt'] is False
    assert result['matwt'] is False

def test_wtcheck_single_value():
    wt = np.array([3])
    result = wtcheck(3, wt)
    expected = 3 * np.ones((3, 1))
    assert result['wtvec'].shape == (3, 1)
    assert np.all(result['wtvec'] == expected)
    assert result['onewt'] is False
    assert result['matwt'] is False

def test_wtcheck_matrix():
    wt = np.eye(4)
    result = wtcheck(4, wt)
    assert result['wtvec'].shape == (4, 4)
    assert result['onewt'] is False
    assert result['matwt'] is True

def test_wtcheck_invalid_n():
    with pytest.raises(ValueError, match="n is not an integer."):
        wtcheck(3.5)
    with pytest.raises(ValueError, match="n is less than 1."):
        wtcheck(0)

def test_wtcheck_nan_in_wtvec():
    wt = np.array([1, np.nan, 1])
    with pytest.raises(ValueError, match="WTVEC has NA values."):
        wtcheck(3, wt)

def test_wtcheck_wrong_shape():
    wt = np.array([[1, 1],
                   [1, 1],
                   [1, 1]])
    with pytest.raises(ValueError, match="WTVEC is neither a vector nor a matrix of order n."):
        wtcheck(3, wt)

def test_wtcheck_wrong_length():
    wt = np.array([1, 1])
    with pytest.raises(ValueError, match="WTVEC of wrong length"):
        wtcheck(3, wt)

def test_wtcheck_negative_in_vector():
    wt = np.array([1, -1, 2])
    with pytest.raises(ValueError, match="Values in WTVEC are not positive."):
        wtcheck(3, wt)

def test_wtcheck_matrix_not_positive_definite():
    wt = np.array([[1, 2],
                   [2, 1]])
    with pytest.raises(ValueError, match="Weight matrix is not positive definite."):
        wtcheck(2, wt)

def test_wtcheck_matrix_with_complex_eig():
    wt = np.array([[0, 1],
                   [-1, 0]])
    with pytest.raises(ValueError, match="Weight matrix has complex eigenvalues."):
        wtcheck(2, wt)
