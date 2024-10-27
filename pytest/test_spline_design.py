import pytest
import numpy as np
from scipy.interpolate import BSpline
import pandas as pd
from GENetLib.spline_design import spline_design


def test_spline_design_normal():
    knots = np.linspace(0, 1, 5)
    x = np.linspace(0, 1, 10)
    design = spline_design(knots, x)
    assert design.shape == (10, 4)

def test_spline_design_out_of_range():
    knots = np.linspace(0, 1, 5)
    x = np.linspace(-1, 2, 10)
    with pytest.raises(ValueError):
        spline_design(knots, x)

def test_spline_design_not_enough_knots():
    knots = np.linspace(0, 1, 3)
    x = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        spline_design(knots, x, norder=4)

def test_spline_design_norder_one():
    knots = [0, 1]
    x = np.linspace(0, 1, 10)
    design = spline_design(knots, x, norder=1)
    assert design.shape == (10, 2)

def test_spline_design_outer_ok_true():
    knots = np.linspace(0, 1, 5)
    x = np.linspace(-1, 2, 10)
    design = spline_design(knots, x, outer_ok=True)
    assert design.shape == (10, 4)

def test_spline_design_with_nan():
    knots = np.linspace(0, 1, 5)
    x = np.linspace(0, 1, 10)
    x[5] = np.nan
    design = spline_design(knots, x)
    assert np.isnan(design.iloc[:, -1].iloc[5])

def test_spline_design_knots_equal_norder():
    knots = np.linspace(0, 1, 4)
    x = np.linspace(0, 1, 10)
    design = spline_design(knots, x, norder=4)
    assert design.shape == (10, 4)
