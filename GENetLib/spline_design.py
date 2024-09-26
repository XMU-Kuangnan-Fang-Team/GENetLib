import numpy as np
import pandas as pd
from scipy.interpolate import BSpline


'''Create B-spline design matrix'''

def spline_design(knots, x, norder=4, outer_ok=False):
    nk = len(knots)
    if nk <= 0:
        raise ValueError("must have at least 'norder' knots")
    knots = np.sort(knots)
    degree = norder - 1
    x = np.array(x)
    need_outer = any(x < knots[degree]) or any(x > knots[nk - degree - 1])
    if not outer_ok and need_outer:
        raise ValueError("x must be within the knot range unless outer_ok = True")
    extended_knots = np.concatenate(([knots[0]] * (degree+1), knots, [knots[-1]] * (degree+1)))
    coef = np.eye(nk + degree + 1)
    spl = BSpline(extended_knots, coef, degree)
    if norder == 1:
        design_pre = spl(x)
        design_range = int((design_pre.shape[1]-nk+1) / 2)
        design = pd.DataFrame(design_pre[:,  design_range : - design_range])
        if outer_ok == False and sum(design.iloc[-1]) < 0.5 :
            m, n = design.shape
            design.loc[m-1, n-1] = 1
        return design
    else:
        design = pd.DataFrame(spl(x)[:, degree:-degree]).iloc[:, 1:-1]
        if outer_ok == False and sum(design.iloc[-1]) < 0.5 :
            m, n = design.shape
            design.loc[m-1, n] = 1
        return design

