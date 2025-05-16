import numpy as np
from GENetLib.fda_func import ycheck, ppbspline, wtcheck, vec2lfd, norder_bspline, fdpar, fdparcheck, create_bspline_basis, create_fourier_basis, fd, int2lfd
from GENetLib.BsplineFunc import BsplineFunc
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
    fdParobj = fdpar(fdobj = basisobj, Lfdobj = 2, lambda_ = 0.5, estimate = True)
    assert isinstance(fdParobj, dict)
    assert 'fd' in fdParobj and isinstance(fdParobj['fd'], dict)
    assert 'lfd' in fdParobj and isinstance(fdParobj['lfd'], dict)
    assert 'lambda' in fdParobj and fdParobj['lambda'] == 0.5
    assert 'estimate' in fdParobj and fdParobj['estimate'] == True
    assert 'penmat' in fdParobj
    basisobj = create_fourier_basis(rangeval = [0, 1], nbasis = 3)
    fdobj = fd(coef = np.zeros((3, 3)), basisobj = basisobj)
    fdParobj = fdpar(fdobj = fdobj, Lfdobj = 1, lambda_ = 0.1)
    assert isinstance(fdParobj, dict)
    assert 'fd' in fdParobj and isinstance(fdParobj['fd'], dict)
    assert 'lfd' in fdParobj and isinstance(fdParobj['lfd'], dict)
    assert 'lambda' in fdParobj and fdParobj['lambda'] == 0.1
    assert 'estimate' in fdParobj and isinstance(fdParobj['estimate'], bool)
    assert 'penmat' in fdParobj

def test_fdparcheck():
    basisobj = create_bspline_basis(rangeval = [0, 1], nbasis = 4, norder = 3)
    fdParobj = fdpar(fdobj = basisobj)
    checked_fdParobj = fdparcheck(fdParobj)
    assert isinstance(checked_fdParobj, dict)
    assert 'fd' in checked_fdParobj and isinstance(checked_fdParobj['fd'], dict)
    assert 'lfd' in checked_fdParobj and isinstance(checked_fdParobj['lfd'], dict)
    assert 'lambda' in checked_fdParobj
    assert 'estimate' in checked_fdParobj and isinstance(checked_fdParobj['estimate'], bool)
    assert 'penmat' in checked_fdParobj
    basisobj = create_fourier_basis(rangeval = [0, 1], nbasis = 3)
    fdobj = fd(coef = np.zeros((3, 3)), basisobj = basisobj)
    fdParobj = fdpar(fdobj = fdobj)
    checked_fdParobj = fdparcheck(fdParobj, ncurve = 5)
    assert isinstance(checked_fdParobj, dict)
    assert 'fd' in checked_fdParobj and isinstance(checked_fdParobj['fd'], dict)
    assert 'lfd' in checked_fdParobj and isinstance(checked_fdParobj['lfd'], dict)
    assert 'lambda' in checked_fdParobj
    assert 'estimate' in checked_fdParobj and isinstance(checked_fdParobj['estimate'], bool)
    assert 'penmat' in checked_fdParobj
    invalid_fdParobj = "Invalid fdParobj"
    try:
        fdparcheck(invalid_fdParobj)
        assert False, "Expected AssertionError for invalid fdParobj type"
    except AssertionError:
        assert True
    basisobj = create_bspline_basis(rangeval = [0, 1], nbasis = 4, norder = 3)
    fdParobj = fdpar(fdobj = basisobj)
    try:
        fdparcheck(fdParobj, ncurve = 5)
        assert False, "Expected AssertionError for basis type fdParobj with ncurve"
    except AssertionError:
        assert True

def test_penalty_matrix():
    basisobj = create_bspline_basis()
    bspline_func = BsplineFunc(basisobj)
    penalty_matrix_spline = bspline_func.penalty_matrix(btype='spline')
    assert penalty_matrix_spline.shape == (4, 4), "Penalty matrix shape mismatch for spline type"
    basisobj_fourier = create_fourier_basis()
    bspline_func_fourier = BsplineFunc(basisobj_fourier, Lfdobj=1)
    penalty_matrix_fourier = bspline_func_fourier.penalty_matrix(btype='fourier')
    assert penalty_matrix_fourier.shape == (3, 3), "Penalty matrix shape mismatch for fourier type"
    try:
        bspline_func.penalty_matrix(btype='invalid_type')
    except ValueError as e:
        assert str(e) == "Wrong basis type", "Invalid btype did not raise ValueError or message incorrect"

def test_smooth_basis():
    basisobj = create_bspline_basis()
    bspline_func = BsplineFunc(basisobj)
    argvals = np.linspace(0, 1, 10)
    y = np.random.rand(10, 5)
    smooth_result = bspline_func.smooth_basis(argvals, y)
    expected_keys = ['fd', 'df', 'gcv', 'beta', 'SSE', 'penmat', 'y2cMap', 'argvals', 'y']
    for key in expected_keys:
        assert key in smooth_result, f"Missing key '{key}' in smooth_result"
    assert isinstance(smooth_result['fd'], dict), "'fd' is not an instance of fd"
    smooth_result_wt = bspline_func.smooth_basis(argvals, y)
    for key in expected_keys:
        assert key in smooth_result_wt, f"Missing key '{key}' in smooth_result_wt"
    smooth_result_cov = bspline_func.smooth_basis(argvals, y)
    for key in expected_keys:
        assert key in smooth_result_cov, f"Missing key '{key}' in smooth_result_cov"
    try:
        bspline_func.smooth_basis(argvals, np.random.rand(10, 6))
    except ValueError as e:
        assert str(e) == "The number of basis functions = 5 exceeds 10 = the number of points to be smoothed.", "Invalid input dimensions did not raise ValueError or message incorrect"

def test_BsplineFunc_initialization():
    basisobj = create_bspline_basis()
    bspline_func = BsplineFunc(basisobj)
    assert hasattr(bspline_func, 'basisobj'), "Missing 'basisobj' attribute"
    assert hasattr(bspline_func, 'Lfdobj'), "Missing 'Lfdobj' attribute"
    assert hasattr(bspline_func, 'rng'), "Missing 'rng' attribute"
    assert hasattr(bspline_func, 'returnMatrix'), "Missing 'returnMatrix' attribute"
    Lfdobj = 3
    rng = [0.1, 0.9]
    bspline_func_custom = BsplineFunc(basisobj, Lfdobj=Lfdobj, rng=rng)
    assert bspline_func_custom.Lfdobj == 3, "Lfdobj not set correctly"
    assert np.array_equal(bspline_func_custom.rng, np.array([0.1, 0.9])), "rng not set correctly"
    try:
        BsplineFunc({'nbasis': 'invalid', 'params': [0.25, 0.5, 0.75], 'rangeval': [0, 1]})
    except ValueError as e:
        assert str(e) == "Value of 'lambda' was negative. 0 used instead.", "Invalid basisobj did not raise ValueError or message incorrect"

def test_BsplineFunc_methods():
    basisobj = create_bspline_basis()
    bspline_func = BsplineFunc(basisobj)
    penalty_matrix = bspline_func.penalty_matrix()
    assert penalty_matrix.shape[0] == penalty_matrix.shape[1], "Penalty matrix is not square"
    argvals = np.linspace(0, 1, 10)
    y = np.random.rand(10, 5)
    smooth_result = bspline_func.smooth_basis(argvals, y)
    expected_keys = ['fd', 'df', 'gcv', 'beta', 'SSE', 'penmat', 'y2cMap', 'argvals', 'y']
    for key in expected_keys:
        assert key in smooth_result, f"Missing key '{key}' in smooth_result"

def test_BsplineFunc_integration():
    basisobj = create_bspline_basis()
    bspline_func = BsplineFunc(basisobj)
    argvals = np.linspace(0, 1, 10)
    y = np.random.rand(10, 5)
    penalty_matrix = bspline_func.penalty_matrix()
    smooth_result = bspline_func.smooth_basis(argvals, y)
    assert not np.array_equal(penalty_matrix, smooth_result['penmat']), "Penalty matrix not used in smooth_result"
    smooth_result_wt_cov = bspline_func.smooth_basis(argvals, y)
    expected_keys = ['fd', 'df', 'gcv', 'beta', 'SSE', 'penmat', 'y2cMap', 'argvals', 'y']
    for key in expected_keys:
        assert key in smooth_result_wt_cov, f"Missing key '{key}' in smooth_result_wt_cov"
 
