import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from GENetLib.plot_gene import plot_fd
from GENetLib.fda_func import fd_chk
from GENetLib.fda_func import basis_fd
from GENetLib.plot_gene import plot_rawdata
from GENetLib.sim_data import sim_data_func
from GENetLib.fda_func import dense_to_func


def test_plot_fd():
    plot_fd(fd_chk(basis_fd(btype="fourier", rangeval=[0, 1], nbasis=4, params=[2 * np.pi]))[1])
    assert plt.gcf() is not None
    func_continuous = sim_data_func(2, 20, 'Continuous', seed=111)
    location = list(func_continuous['location'])
    X = func_continuous['X']
    dense_to_func_res = dense_to_func(location, X, btype = "Bspline", nbasis = 5, params = 4, Plot = False)
    plot_fd(dense_to_func_res, xlab = 'Location', ylab = 'X', title="plot_fd test", colors=['red', 'green'],
            linestyles=['-', '--'], legend=['Rep 1', 'Rep 2'], grid=True, figsize=(10, 6))
    assert plt.gcf() is not None

def test_plot_rawdata():
    func_continuous = sim_data_func(n=1500, m=30, ytype='Continuous', seed=123)
    plot_rawdata(func_continuous['location'], func_continuous['X'], color = None, pch = 4, cex = 0.9)
    assert plt.gcf() is not None
    plot_rawdata(func_continuous['location'], func_continuous['X'], color = 'green', pch = 4, cex = 0.9)
    assert plt.gcf() is not None
    location = np.array([0, 1, 2, 3, 4])
    X = np.array([[1, np.nan, 3, np.nan, 5], [5, 4, np.nan, 2, np.nan]])
    plot_rawdata(location, X, color='red', show_legend = False, title = 'plot_rawdata test',
                 grid = True, figsize = (10,6)))
    assert plt.gcf() is not None
    
