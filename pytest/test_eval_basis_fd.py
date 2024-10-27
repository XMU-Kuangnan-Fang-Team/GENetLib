import pytest
import numpy as np
from GENetLib.eval_basis_fd import eval_basis, eval_fd
from GENetLib.create_basis import create_bspline_basis

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
    assert isinstance(basis_values, np.matrix)

# Test eval_fd function
def test_eval_fd():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4)
    fdobj = {'coefs': coef, 'basis': basisobj}
    evalarg = np.linspace(0, 1, 10)
    function_values = eval_fd(evalarg, fdobj)
    assert function_values.shape == (10,)

def test_eval_fd_derivative():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4)
    fdobj = {'coefs': coef, 'basis': basisobj}
    evalarg = np.linspace(0, 1, 10)
    function_values = eval_fd(evalarg, fdobj, Lfdobj=1)
    assert function_values.shape == (10,)

def test_eval_fd_return_matrix():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4)
    fdobj = {'coefs': coef, 'basis': basisobj}
    evalarg = np.linspace(0, 1, 10)
    function_values = eval_fd(evalarg, fdobj, returnMatrix=True)
    assert isinstance(function_values, np.matrix)

# Test error handling in eval_fd
def test_eval_fd_error():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4)
    fdobj = {'coefs': coef, 'basis': basisobj}
    evalarg = np.array([0, 2])  # Out of range values
    with pytest.raises(ValueError):
        eval_fd(evalarg, fdobj)

# Test eval_fd with multidimensional coefficients
def test_eval_fd_multidimensional():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4, 2, 3)  # Multidimensional coefficients
    fdobj = {'coefs': coef, 'basis': basisobj}
    evalarg = np.linspace(0, 1, 10)
    function_values = eval_fd(evalarg, fdobj)
    assert function_values.shape == (10, 2, 3)

# Test eval_fd with list input
def test_eval_fd_list_input():
    basisobj = create_bspline_basis(rangeval=[0, 1], nbasis=4, norder=4)
    coef = np.random.rand(4)
    fdobj = {'coefs': coef, 'basis': basisobj}
    evalarg = list(np.linspace(0, 1, 10))
    function_values = eval_fd(evalarg, fdobj)
    assert function_values.shape == (10,)
