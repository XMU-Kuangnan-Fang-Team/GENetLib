import numpy as np
from GENetLib.fda_func import ycheck
import pytest

# 测试 ycheck 函数
def test_ycheck():
    # 测试输入为一维数组
    y = np.array([1, 2, 3, 4])
    n = 4
    result = ycheck(y, n)
    assert result['y'].shape == (4,)
    assert result['ncurve'] == 1
    assert result['nvar'] == 1
    assert result['ndim'] == 1

    # 测试输入为二维数组
    y = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    n = 4
    result = ycheck(y, n)
    assert result['y'].shape == (4, 2)
    assert result['ncurve'] == 2
    assert result['nvar'] == 1
    assert result['ndim'] == 2

    # 测试输入为三维数组
    y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    n = 4
    result = ycheck(y, n)
    assert result['y'].shape == (4, 2, 2)
    assert result['ncurve'] == 2
    assert result['nvar'] == 2
    assert result['ndim'] == 3

    # 测试输入为列表
    y = [1, 2, 3, 4]
    n = 4
    result = ycheck(y, n)
    assert result['y'].shape == (4,)
    assert result['ncurve'] == 1
    assert result['nvar'] == 1
    assert result['ndim'] == 1

    # 测试输入不是数组或列表
    y = "not an array"
    n = 4
    with pytest.raises(ValueError) as excinfo:
        ycheck(y, n)
    assert "Y is not of class matrix or class array." in str(excinfo.value)

    # 测试输入数组的第一维长度不等于 n
    y = np.array([1, 2, 3])
    n = 4
    with pytest.raises(ValueError) as excinfo:
        ycheck(y, n)
    assert "Y is not the same length as ARGVALS." in str(excinfo.value)

    # 测试输入数组的维度超过3
    y = np.random.rand(4, 2, 2, 2)
    n = 4
    with pytest.raises(ValueError) as excinfo:
        ycheck(y, n)
    assert "Second argument must not have more than 3 dimensions" in str(excinfo.value)