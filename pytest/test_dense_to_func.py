import pytest
import numpy as np
from GENetLib.dense_to_func import dense_to_func
from GENetLib.create_basis import create_bspline_basis, create_expon_basis, create_fourier_basis, create_monomial_basis, create_power_basis


def test_dense_to_func_bspline():
    location = np.linspace(0, 1, 100)
    X = np.sin(2 * np.pi * location) + np.random.normal(0, 0.1, size=(100, 1))
    tofunc = dense_to_func(location.tolist(), X, "Bspline", 4, 3)
    assert 'coefs' in tofunc

def test_dense_to_func_exponential():
    location = np.linspace(0, 2, 100)
    X = np.exp(-location) + np.random.normal(0, 0.1, size=(100, 1))
    tofunc = dense_to_func(location.tolist(), X, "Exponential", 1, [0.5])
    assert 'coefs' in tofunc

def test_dense_to_func_fourier():
    location = np.linspace(0, 2 * np.pi, 100)
    X = np.sin(location) + np.random.normal(0, 0.1, size=(100, 1))
    tofunc = dense_to_func(location.tolist(), X, "Fourier", 4, 2 * np.pi)
    assert 'coefs' in tofunc

def test_dense_to_func_monomial():
    location = np.linspace(0, 1, 100)
    X = location**2 + np.random.normal(0, 0.1, size=(100, 1))
    tofunc = dense_to_func(location.tolist(), X, "Monomial", 1, [2])
    assert 'coefs' in tofunc

def test_dense_to_func_power():
    location = np.linspace(0, 1, 100)
    X = location**2 + np.random.normal(0, 0.1, size=(100, 1))
    tofunc = dense_to_func(location.tolist(), X, "Power", 1, [2])
    assert 'coefs' in tofunc


def test_dense_to_func_plot():
    location = np.linspace(0, 1, 100)
    X = np.sin(2 * np.pi * location) + np.random.normal(0, 0.1, size=(100, 1))
    tofunc = dense_to_func(location.tolist(), X, "Bspline", 4, 3, Plot=True)
    assert 'coefs' in tofunc

def test_dense_to_func_empty():
    location = []
    X = np.empty((0, 1))
    with pytest.raises(ValueError):
        dense_to_func(location, X, "Bspline", 4, 3)  # location cannot be empty

def test_dense_to_func_missing_data():
    # Create a list of locations
    location = np.linspace(0, 10, 100).tolist()
    X = np.random.rand(100, 1)  # Initial matrix with random values
    X[X < 0.5] = np.nan  # Set some values to NaN
    btype = "Bspline"
    nbasis = 4
    params = 4  # Order of the B-spline basis
    tofunc = dense_to_func(location, X, btype, nbasis, params)
    assert 'coefs' in tofunc
