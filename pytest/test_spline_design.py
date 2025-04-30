import numpy as np
from GENetLib.fda_func import spline_design

def test_spline_design_norder_one():
    knots = [0, 1]
    x = np.linspace(0, 1, 10)
    design = spline_design(knots, x, norder=1)
    assert design.shape == (10, 1)
