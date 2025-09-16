import pytest
import numpy as np
from GENetLib.fda_func import eval_basis, eval_fd, lfd, int2lfd
from GENetLib.fda_func import create_bspline_basis


# Test eval_basis function
def test_eval_basis():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    evalarg = np.linspace(0, 1, 10)
    basis_values = eval_basis(evalarg, basisobj)
    assert basis_values.shape == (10, 4)


def test_eval_basis_derivative():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    evalarg = np.linspace(0, 1, 10)
    basis_values = eval_basis(evalarg, basisobj, Lfdobj=1)
    assert basis_values.shape == (10, 4)


def test_eval_basis_return_matrix():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    evalarg = np.linspace(0, 1, 10)
    basis_values = eval_basis(evalarg, basisobj, returnMatrix=True)
    assert isinstance(basis_values, np.ndarray)


# Test eval_fd function
def test_eval_fd():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4)
    fdobj = {"coefs": coef, "basis": basisobj}
    evalarg = np.linspace(0, 1, 10)
    function_values = eval_fd(evalarg, fdobj)
    assert function_values.shape == (10, 1)


def test_eval_fd_derivative():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4)
    fdobj = {"coefs": coef, "basis": basisobj}
    evalarg = np.linspace(0, 1, 10)
    function_values = eval_fd(evalarg, fdobj, Lfdobj=1)
    assert function_values.shape == (10, 1)


def test_eval_fd_return_matrix():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4)
    fdobj = {"coefs": coef, "basis": basisobj}
    evalarg = np.linspace(0, 1, 10)
    function_values = eval_fd(evalarg, fdobj, returnMatrix=True)
    assert function_values is not None


# Test eval_fd with multidimensional coefficients
def test_eval_fd_multidimensional():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4, 2, 3)  # Multidimensional coefficients
    fdobj = {"coefs": coef, "basis": basisobj}
    evalarg = np.linspace(0, 1, 10)
    function_values = eval_fd(evalarg, fdobj)
    assert function_values.shape == (10, 2, 3)


# Test eval_fd with list input
def test_eval_fd_list_input():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4)
    fdobj = {"coefs": coef, "basis": basisobj}
    evalarg = list(np.linspace(0, 1, 10))
    function_values = eval_fd(evalarg, fdobj)
    if function_values.ndim == 1:
        function_values = function_values[np.newaxis, :]
    assert function_values.shape == (1, 10)


# Test lfd
def test_lfd():
    function_values = lfd()
    assert function_values is not None


# Test int2lfd
def test_int2lfd():
    function_values = int2lfd()
    assert function_values is not None
