import pytest
import numpy as np
import math

from GENetLib.fda_func import (
    create_bspline_basis,
    create_expon_basis,
    create_fourier_basis,
    create_monomial_basis,
    create_power_basis,
    create_constant_basis,
)

# Test B-spline basis
def test_create_bspline_basis():
    basis = create_bspline_basis(rangeval=[0, 1], nbasis=5, breaks=[0, 0.5, 1])
    assert 'names' in basis
    assert basis['btype'] == 'bspline'

def test_create_bspline_basis_errors():
    with pytest.raises(ValueError):
        create_bspline_basis(breaks=[1, 0])  # Decreasing breaks
    with pytest.raises(ValueError):
        create_bspline_basis(breaks=[0, 0])  # Non-unique breaks
    with pytest.raises(ValueError):
        create_bspline_basis(rangeval=[1,1])  # Invalid rangeval

# Test Exponential basis
def test_create_expon_basis():
    basis = create_expon_basis(rangeval=[0, 1], nbasis=2, ratevec=[0.5, 1.5])
    assert 'names' in basis
    assert basis['btype'] == 'expon'

# Test Fourier basis
def test_create_fourier_basis():
    basis = create_fourier_basis(rangeval=[0, 1], nbasis=3)
    assert 'names' in basis
    assert basis['btype'] == 'fourier'

def test_create_fourier_basis_errors():
    with pytest.raises(ValueError):
        create_fourier_basis(period=-1)  # Negative period

# Test Monomial basis
def test_create_monomial_basis():
    basis = create_monomial_basis(rangeval=[0, 1], nbasis=3, exponents=[0, 1, 2])
    assert 'names' in basis
    assert basis['btype'] == 'monom'

def test_create_monomial_basis_errors():
    with pytest.raises(ValueError):
        create_monomial_basis(exponents=[0, 1, 2, 2])  # Duplicate exponents

# Test Power basis
def test_create_power_basis():
    basis = create_power_basis(rangeval=[0, 1], nbasis=3, exponents=[0, 1, 2])
    assert 'names' in basis
    assert basis['btype'] == 'power'

def test_create_power_basis_errors():
    with pytest.raises(ValueError):
        create_power_basis(exponents=[0, 1, 2, 2])  # Duplicate exponents

# Test Constant basis
def test_create_constant_basis():
    basis = create_constant_basis()
    assert 'names' in basis
    assert basis['btype'] == 'const'

# Additional tests for edge cases and specific conditions
def test_create_bspline_basis_edge_cases():
    basis = create_bspline_basis(rangeval=[0, 1], nbasis=4)
    assert basis['nbasis'] == 4

def test_create_expon_basis_edge_cases():
    basis = create_expon_basis(rangeval=[0, 1], nbasis=1)
    assert basis['nbasis'] == 1

def test_create_fourier_basis_edge_cases():
    basis = create_fourier_basis(rangeval=[0, 1], nbasis=1)
    assert basis['nbasis'] == 1

def test_create_monomial_basis_edge_cases():
    basis = create_monomial_basis(rangeval=[0, 1], nbasis=1)
    assert basis['nbasis'] == 1

def test_create_power_basis_edge_cases():
    basis = create_power_basis(rangeval=[0, 1], nbasis=1)
    assert basis['nbasis'] == 1

def test_create_constant_basis_edge_cases():
    basis = create_constant_basis()
    assert basis['nbasis'] == 1
