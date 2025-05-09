import numpy as np
from GENetLib.fda_func import ycheck
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
