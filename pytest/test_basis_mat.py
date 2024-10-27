import pytest
import numpy as np
import pandas as pd
from GENetLib.spline_design import spline_design

# Test B-spline matrix
def test_bspline_mat():
    x = np.linspace(0, 1, 10)
    breaks = np.linspace(0, 1, 5)
    result = bspline_mat(x, breaks)
    assert result.shape == (10, 7)

def test_bspline_mat_errors():
    x = np.linspace(0, 1, 10)
    breaks = [0, 0.5]
    with pytest.raises(ValueError):
        bspline_mat(x, breaks)  # Not enough breaks
    with pytest.raises(ValueError):
        bspline_mat(x, breaks, norder=0)  # Invalid order
    with pytest.raises(ValueError):
        bspline_mat(x, breaks, nderiv=-1)  # Negative derivative

# Test Exponential matrix
def test_expon_mat():
    x = np.linspace(0, 1, 10)
    ratevec = [0.5]
    result = expon_mat(x, ratevec)
    assert result.shape == (10, 1)

def test_expon_mat_errors():
    x = np.linspace(0, 1, 10)
    ratevec = [0]  # Non-positive rate
    with pytest.raises(ValueError):
        expon_mat(x, ratevec)

# Test Fourier matrix
def test_fourier_mat():
    x = np.linspace(0, 1, 10)
    result = fourier_mat(x)
    assert result.shape[1] > 1

def test_fourier_mat_errors():
    x = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        fourier_mat(x, period=-1)  # Negative period
    with pytest.raises(ValueError):
        fourier_mat(x, nbasis=-1)  # Negative basis

# Test Monomial matrix
def test_monomial_mat():
    x = np.linspace(0, 1, 10)
    exponents = [1, 2]
    result = monomial_mat(x, exponents)
    assert result.shape == (10, 2)

def test_monomial_mat_errors():
    x = np.linspace(0, 1, 10)
    exponents = [1.5, 2]  # Non-integer exponent
    with pytest.raises(ValueError):
        monomial_mat(x, exponents)
    exponents = [1, 1]  # Duplicate exponents
    with pytest.raises(ValueError):
        monomial_mat(x, exponents)

# Test Polynomial matrix
def test_polyg_mat():
    x = np.linspace(0, 1, 10)
    argvals = np.linspace(0, 1, 5)
    result = polyg_mat(x, argvals)
    assert result.shape == (10, 4)

def test_polyg_mat_errors():
    x = np.linspace(0, 1, 10)
    argvals = [0, 0.5, 0.5]  # Non-strictly increasing argvals
    with pytest.raises(ValueError):
        polyg_mat(x, argvals)

# Test Power matrix
def test_power_mat():
    x = np.linspace(0, 1, 10)
    exponents = [1, 2]
    result = power_mat(x, exponents)
    assert result.shape == (10, 2)

def test_power_mat_errors():
    x = np.linspace(0, 1, 10)
    exponents = [1, -1]  # Negative exponent
    with pytest.raises(ValueError):
        power_mat(x, exponents)
    exponents = [2, 3]
    with pytest.raises(ValueError):
        power_mat(x, exponents, nderiv=3)  # Derivative exceeds exponent

# Additional tests for edge cases and specific conditions
def test_bspline_mat_return_matrix():
    x = np.linspace(0, 1, 10)
    breaks = np.linspace(0, 1, 5)
    result = bspline_mat(x, breaks, returnMatrix=True)
    assert isinstance(result, np.ndarray)

def test_bspline_mat_nderiv():
    x = np.linspace(0, 1, 10)
    breaks = np.linspace(0, 1, 5)
    result = bspline_mat(x, breaks, nderiv=1)
    assert result.shape == (10, 7)

def test_fourier_mat_even_nbasis():
    x = np.linspace(0, 1, 10)
    result = fourier_mat(x, nbasis=4)
    assert result.shape[1] == 5  # Should be odd

def test_monomial_mat_nderiv():
    x = np.linspace(0, 1, 10)
    exponents = [2, 3]
    result = monomial_mat(x, exponents, nderiv=1)
    assert result.shape == (10, 2)

def test_power_mat_zero_exponent():
    x = np.array([0, 1, 2])
    exponents = [0, 1]
    with pytest.raises(ValueError):
        power_mat(x, exponents)
